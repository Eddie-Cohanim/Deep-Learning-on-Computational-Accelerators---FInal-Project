# Note: This implementation uses plain convolutional blocks without skip connections.
# If training accuracy plateaus or gradients vanish in deeper configurations,
# consider upgrading the feature extractor to use residual blocks (skip connections),
# which add the input of each block directly to its output and significantly improve
# gradient flow in deep networks.

import pathlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Sequence

from model.Augmentations.augmentations import (
    HorizontalFlipAugmentation,
    RotationAugmentation,
    ColorJitterAugmentation,
    GaussianBlurAugmentation,
    PerspectiveAugmentation,
    GammaAugmentation,
)
from model.augmented_dataset import AugmentedTrainingDataset


_SUPPORTED_POOLING_TYPES = {
    "max": nn.MaxPool2d,
    "avg": nn.AvgPool2d,
}

_SUPPORTED_ACTIVATIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    # Softmax is intentionally excluded: it normalises outputs to sum to 1,
    # which destroys relative magnitude information and kills gradient flow
    # when used as a hidden-layer activation. Use it only at inference time
    # on the final output, not as an intermediate activation.
}


class CNN(nn.Module):
    """
    A convolutional neural network for multi-class image classification.

    Architecture:
        [(Conv2d -> BatchNorm2d -> Activation) * pool_every -> MaxPool2d] * groups
        -> Flatten
        -> [Linear -> Activation -> Dropout] * len(hidden_dims)
        -> Linear(num_classes)

    All hyperparameters and training settings are fixed at construction time.
    The training, validation, and test methods accept only a path to a folder
    of images organized as: path/<class_name>/<image_file>
    """

    def __init__(
        self,
        in_size: tuple,
        class_names: list,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        num_epochs: int,
        optimizer_class: type,
        loss_function: nn.Module,
        conv_kernel_size: int,
        pooling_type: str,
        pool_kernel_size: int,
        image_normalization_mean: list,
        image_normalization_std: list,
        num_dataloader_workers: int,
        batch_size: int,
        activation: str,
        use_batchnorm: bool,
        dropout_probability: float,
        learning_rate: float,
        weight_decay: float,
        early_stopping_patience: int,
        number_of_augmented_copies_per_image: int,
        run_augmentation: bool,
        augmentation_rotation_max_degrees: float,
        augmentation_brightness_jitter: float,
        augmentation_contrast_jitter: float,
        augmentation_saturation_jitter: float,
    ):
        """
        :param in_size: Shape of a single input image as (channels, height, width).
            Images from disk will be resized to match (height, width).
        :param class_names: Ordered list of class label strings, e.g.
            ["beer_lager", "beer_stout", "wine_red"]. The index of each name
            corresponds to the integer label the model outputs.
        :param channels: Output channel count for each convolutional layer.
            Length determines the total number of conv layers.
        :param pool_every: Number of conv layers between each max-pooling step.
        :param hidden_dims: Output dimension for each fully-connected hidden layer.
        :param num_epochs: Number of full passes over the training set during training.
        :param optimizer_class: The optimizer class to use, e.g. torch.optim.Adam,
            torch.optim.SGD, or torch.optim.AdamW. Must be provided — no default.
        :param loss_function: An instantiated loss function, e.g. nn.CrossEntropyLoss().
            Must be provided — no default.
        :param conv_kernel_size: Kernel size for all convolutional layers. Padding is
            set automatically to preserve spatial dimensions (kernel_size // 2).
        :param pooling_type: Type of pooling to apply. Either 'max' or 'avg'.
        :param pool_kernel_size: Kernel size for pooling layers.
        :param image_normalization_mean: Per-channel mean for input normalization.
            Defaults to ImageNet mean [0.485, 0.456, 0.406].
        :param image_normalization_std: Per-channel std for input normalization.
            Defaults to ImageNet std [0.229, 0.224, 0.225].
        :param num_dataloader_workers: Number of worker processes for loading images.
            0 means images are loaded in the main process.
        :param batch_size: Number of images per batch when loading from disk.
        :param activation: Activation function to use. Either 'relu', 'lrelu',
            'tanh', 'sigmoid', or 'softmax'.
        :param use_batchnorm: Whether to apply batch normalization after each conv layer.
        :param dropout_probability: Dropout probability applied after each hidden FC layer.
            Set to 0.0 to disable dropout.
        :param learning_rate: Learning rate passed to the optimizer at construction.
        :param weight_decay: L2 regularization coefficient passed to the optimizer.
            Effective with AdamW; has no effect when using plain Adam.
        :param early_stopping_patience: Number of epochs without improvement in validation
            loss before training is stopped early. Set to 0 to disable early stopping.
        :param number_of_augmented_copies_per_image: Number of additional augmented versions
            to generate for each training image on the fly. The effective training set size
            is multiplied by (1 + number_of_augmented_copies_per_image).
        :param run_augmentation: Whether to apply online augmentation during training.
            When False, training images are loaded as-is with no augmentation applied.
        :param augmentation_rotation_max_degrees: Maximum rotation magnitude in degrees for
            online augmentation. The angle is sampled uniformly from
            [-max, +max] for each augmented copy.
        :param augmentation_brightness_jitter: Maximum fractional change in brightness for
            online augmentation.
        :param augmentation_contrast_jitter: Maximum fractional change in contrast for
            online augmentation.
        :param augmentation_saturation_jitter: Maximum fractional change in saturation for
            online augmentation.
        """
        super().__init__()

        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from: {list(_SUPPORTED_ACTIVATIONS.keys())}"
            )

        if pooling_type not in _SUPPORTED_POOLING_TYPES:
            raise ValueError(
                f"Unsupported pooling type '{pooling_type}'. "
                f"Choose from: {list(_SUPPORTED_POOLING_TYPES.keys())}"
            )

        self._class_names = list(class_names)
        self._in_size = in_size
        self._num_classes = len(class_names)
        self._channels = list(channels)
        self._pool_every = pool_every
        self._hidden_dims = list(hidden_dims)
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._activation = activation
        self._pooling_type = pooling_type
        self._pool_kernel_size = pool_kernel_size
        self._conv_kernel_size = conv_kernel_size
        self._use_batchnorm = use_batchnorm
        self._dropout_probability = dropout_probability
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._optimizer_class = optimizer_class
        self._num_dataloader_workers = num_dataloader_workers
        self._early_stopping_patience = early_stopping_patience
        self._number_of_augmented_copies_per_image = number_of_augmented_copies_per_image
        self._run_augmentation = run_augmentation

        self._online_augmentation_sequence = [
            HorizontalFlipAugmentation(),
            RotationAugmentation(max_rotation_degrees=augmentation_rotation_max_degrees),
            ColorJitterAugmentation(
                brightness_jitter=augmentation_brightness_jitter,
                contrast_jitter=augmentation_contrast_jitter,
                saturation_jitter=augmentation_saturation_jitter,
            ),
            GaussianBlurAugmentation(),
            PerspectiveAugmentation(),
            GammaAugmentation(),
        ]

        _, image_height, image_width = in_size
        self._image_transforms = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std),
        ])

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._feature_extractor = self._build_feature_extractor()
        self._classifier_head = self._build_classifier_head()
        self.to(self._device)
        self._optimizer = optimizer_class(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._loss_function = loss_function
        self._grad_scaler = torch.cuda.amp.GradScaler(enabled=self._device.type == "cuda")

    def _load_dataset(self, dataset_path: pathlib.Path, shuffle: bool) -> DataLoader:
        """
        Loads images from a folder using the structure:
            dataset_path/<class_name>/<image_file>

        :param dataset_path: Root folder containing one subfolder per class.
        :param shuffle: Whether to shuffle the images on each iteration.
        :return: A DataLoader over the images in the given folder.
        """
        dataset = ImageFolder(root=str(dataset_path), transform=self._image_transforms)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_dataloader_workers > 0,
        )

    def _load_augmented_training_dataset(self, dataset_path: pathlib.Path) -> DataLoader:
        """
        Loads the training set as an augmented dataset that exposes
        (1 + number_of_augmented_copies_per_image) versions of each image.

        The original copy receives only the base transform (resize + normalize).
        Each augmented copy is passed through the full online augmentation
        sequence — horizontal flip, rotation, color jitter, gaussian blur —
        before the base transform is applied. The augmentations are applied in
        memory; nothing is written to disk.

        :param dataset_path: Root folder containing one subfolder per class.
        :return: A DataLoader over the expanded augmented training dataset.
        """
        base_dataset = ImageFolder(root=str(dataset_path), transform=None)
        augmented_training_dataset = AugmentedTrainingDataset(
            base_dataset=base_dataset,
            base_transform=self._image_transforms,
            augmentation_sequence=self._online_augmentation_sequence,
            number_of_augmented_copies=self._number_of_augmented_copies_per_image,
        )
        return DataLoader(
            augmented_training_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_dataloader_workers > 0,
        )

    def _build_feature_extractor(self) -> nn.Sequential:
        """
        Builds the convolutional feature extraction layers.

        Pattern: [(Conv2d -> BatchNorm2d -> Activation) * pool_every -> MaxPool2d] * groups
        Remaining conv layers (when len(channels) is not divisible by pool_every)
        appear at the end without a pooling step.
        """
        activation_class = _SUPPORTED_ACTIVATIONS[self._activation]
        input_channels = self._in_size[0]
        layers = []

        pooling_class = _SUPPORTED_POOLING_TYPES[self._pooling_type]
        conv_padding = self._conv_kernel_size // 2

        for conv_index, output_channels in enumerate(self._channels, start=1):
            layers.append(nn.Conv2d(
                input_channels, output_channels,
                kernel_size=self._conv_kernel_size,
                padding=conv_padding,
            ))
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(output_channels))
            layers.append(activation_class())
            input_channels = output_channels

            if conv_index % self._pool_every == 0:
                layers.append(pooling_class(kernel_size=self._pool_kernel_size))

        return nn.Sequential(*layers)

    def _compute_feature_output_size(self) -> int:
        """
        Determines the flattened size of the feature extractor output by running
        a single dummy tensor through it.

        The random number generator state is saved and restored so that this
        probe does not affect any random operations elsewhere in the program.
        """
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.zeros(1, *self._in_size)
            feature_output = self._feature_extractor(dummy_input)
            flattened_size = feature_output.numel()
        finally:
            torch.set_rng_state(rng_state)
        return flattened_size

    def _build_classifier_head(self) -> nn.Sequential:
        """
        Builds the fully-connected classification head.

        Pattern: [Linear -> Activation -> Dropout] * len(hidden_dims) -> Linear(num_classes)
        """
        activation_class = _SUPPORTED_ACTIVATIONS[self._activation]
        current_input_dim = self._compute_feature_output_size()
        layers = []

        for hidden_dim in self._hidden_dims:
            layers.append(nn.Linear(current_input_dim, hidden_dim))
            layers.append(activation_class())
            if self._dropout_probability > 0.0:
                layers.append(nn.Dropout(p=self._dropout_probability))
            current_input_dim = hidden_dim

        layers.append(nn.Linear(current_input_dim, self._num_classes))
        return nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        """
        Runs a forward pass through the network.

        :param images: Input batch of shape (batch_size, channels, height, width).
        :return: Raw class score tensor of shape (batch_size, num_classes).
        """
        extracted_features = self._feature_extractor(images)
        batch_size = images.shape[0]
        flattened_features = extracted_features.reshape(batch_size, -1)
        class_scores = self._classifier_head(flattened_features)
        return class_scores

    def train_on_dataset(
        self,
        dataset_path: pathlib.Path,
        val_dataset_path: pathlib.Path = None,
    ) -> dict:
        """
        Trains the model on images loaded from the given folder.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        If val_dataset_path is provided and early_stopping_patience > 0, training
        stops early when validation loss has not improved for that many epochs.
        The best weights (lowest validation loss) are restored at the end.

        :param dataset_path: Path to the folder containing class subfolders.
        :param val_dataset_path: Optional path to the validation split, used for
            early stopping. Required when early_stopping_patience > 0.
        :return: Dictionary with keys 'train_loss' and 'train_accuracy',
            each a list of floats with one value per epoch.
        """
        if self._run_augmentation:
            train_loader = self._load_augmented_training_dataset(dataset_path)
        else:
            train_loader = self._load_dataset(dataset_path, shuffle=True)
        val_loader = self._load_dataset(val_dataset_path, shuffle=False) if val_dataset_path is not None else None
        return self.train_on_data_loaders(train_loader, val_loader)

    def train_on_data_loaders(
        self,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader = None,
    ) -> dict:
        """
        Trains the model using pre-built DataLoaders.

        This is the core training implementation. train_on_dataset delegates to
        this method after constructing its DataLoaders from disk paths.

        If val_data_loader is provided and early_stopping_patience > 0, training
        stops early when validation loss has not improved for that many epochs.
        The best weights (lowest validation loss) are restored at the end.

        :param train_data_loader: DataLoader over the training set.
        :param val_data_loader: Optional DataLoader over the validation set, used
            for early stopping. Required when early_stopping_patience > 0.
        :return: Dictionary with keys 'train_loss' and 'train_accuracy',
            each a list of floats with one value per epoch.
        """
        epoch_losses = []
        epoch_accuracies = []

        early_stopping_enabled = self._early_stopping_patience > 0 and val_data_loader is not None
        epochs_without_improvement = 0
        best_val_loss = float("inf")
        best_weights = None

        for epoch_index in range(1, self._num_epochs + 1):
            cumulative_loss = 0.0
            total_correct_predictions = 0
            total_samples = 0

            self.train()
            for images, labels in train_data_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                self._optimizer.zero_grad()
                with torch.amp.autocast(device_type=self._device.type, enabled=self._device.type == "cuda"):
                    class_scores = self.forward(images)
                    batch_loss = self._loss_function(class_scores, labels)
                self._grad_scaler.scale(batch_loss).backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()

                cumulative_loss += batch_loss.item() * images.shape[0]
                predicted_labels = class_scores.argmax(dim=1)
                total_correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += images.shape[0]

            epoch_average_loss = cumulative_loss / total_samples
            epoch_accuracy = total_correct_predictions / total_samples
            epoch_losses.append(epoch_average_loss)
            epoch_accuracies.append(epoch_accuracy)

            print(f"  Epoch [{epoch_index}/{self._num_epochs}]  loss: {epoch_average_loss:.4f}  accuracy: {epoch_accuracy * 100:.2f}%", end="", flush=True)

            if early_stopping_enabled:
                val_results = self.validate_on_data_loader(val_data_loader)
                current_val_loss = val_results["val_loss"]
                current_val_accuracy = val_results["val_accuracy"]
                print(f"  val_loss: {current_val_loss:.4f}  val_accuracy: {current_val_accuracy * 100:.2f}%", end="", flush=True)

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_without_improvement = 0
                    best_weights = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self._early_stopping_patience:
                    print(f"\n  Early stopping triggered after {epoch_index} epochs (no improvement for {self._early_stopping_patience} epochs).")
                    self.load_state_dict({key: value.to(self._device) for key, value in best_weights.items()})
                    break

            print()

        return {"train_loss": epoch_losses, "train_accuracy": epoch_accuracies}

    def validate_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        """
        Evaluates the model on images loaded from the given folder, without updating weights.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys 'val_loss' and 'val_accuracy' as floats.
        """
        val_loader = self._load_dataset(dataset_path, shuffle=False)
        return self.validate_on_data_loader(val_loader)

    def validate_on_data_loader(self, val_data_loader: DataLoader) -> dict:
        """
        Evaluates the model using a pre-built DataLoader, without updating weights.

        This is the core validation implementation. validate_on_dataset delegates
        to this method after constructing its DataLoader from a disk path.

        :param val_data_loader: DataLoader over the validation set.
        :return: Dictionary with keys 'val_loss' and 'val_accuracy' as floats.
        """
        self.eval()
        cumulative_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_data_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                class_scores = self.forward(images)
                batch_loss = self._loss_function(class_scores, labels)
                cumulative_loss += batch_loss.item() * images.shape[0]
                predicted_labels = class_scores.argmax(dim=1)
                total_correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += images.shape[0]

        return {
            "val_loss": cumulative_loss / total_samples,
            "val_accuracy": total_correct_predictions / total_samples,
        }

    def test_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        """
        Evaluates the model on images loaded from the given folder and returns predictions
        along with per-class precision, recall, F1, and sample counts.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys:
            - 'test_accuracy': overall accuracy as a float
            - 'predictions': 1D Tensor of predicted class indices
            - 'per_class_results': dict mapping each class name to its metrics
        """
        dataset = ImageFolder(root=str(dataset_path), transform=self._image_transforms)
        test_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_dataloader_workers > 0,
        )
        class_names_from_dataset = dataset.classes
        number_of_classes = len(class_names_from_dataset)

        self.eval()
        total_correct_predictions = 0
        total_samples = 0
        all_predicted_labels = []
        all_true_labels = []
        all_confidence_scores = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                class_scores = self.forward(images)
                class_probabilities = torch.softmax(class_scores, dim=1)
                predicted_labels = class_scores.argmax(dim=1)
                max_confidence_per_sample = class_probabilities.max(dim=1).values
                all_predicted_labels.append(predicted_labels.cpu())
                all_true_labels.append(labels.cpu())
                all_confidence_scores.append(max_confidence_per_sample.cpu())
                total_correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += images.shape[0]

        all_predicted_labels_tensor = torch.cat(all_predicted_labels)
        all_true_labels_tensor = torch.cat(all_true_labels)
        all_confidence_scores_tensor = torch.cat(all_confidence_scores)

        per_class_results = {}
        for class_index, class_name in enumerate(class_names_from_dataset):
            true_positives = ((all_predicted_labels_tensor == class_index) & (all_true_labels_tensor == class_index)).sum().item()
            false_positives = ((all_predicted_labels_tensor == class_index) & (all_true_labels_tensor != class_index)).sum().item()
            false_negatives = ((all_predicted_labels_tensor != class_index) & (all_true_labels_tensor == class_index)).sum().item()
            true_negatives = total_samples - true_positives - false_positives - false_negatives
            total_class_samples = true_positives + false_negatives

            predicted_as_this_class_mask = all_predicted_labels_tensor == class_index
            if predicted_as_this_class_mask.sum().item() > 0:
                mean_confidence = all_confidence_scores_tensor[predicted_as_this_class_mask].mean().item()
            else:
                mean_confidence = 0.0

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / total_class_samples if total_class_samples > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_results[class_name] = {
                "total_samples_tested": total_class_samples,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
                "mean_confidence": round(mean_confidence, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1_score, 4),
            }

        raw_predictions = [
            {
                "ground_truth": class_names_from_dataset[true_label_index],
                "predicted": class_names_from_dataset[predicted_label_index],
                "confidence": round(confidence_score, 4),
            }
            for true_label_index, predicted_label_index, confidence_score in zip(
                all_true_labels_tensor.tolist(),
                all_predicted_labels_tensor.tolist(),
                all_confidence_scores_tensor.tolist(),
            )
        ]

        return {
            "test_accuracy": total_correct_predictions / total_samples,
            "per_class_results": per_class_results,
            "raw_predictions": raw_predictions,
        }

    def save_checkpoint(self, file_path: pathlib.Path):
        """
        Saves the current model weights to a file.

        :param file_path: Destination path for the checkpoint file.
        """
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path: pathlib.Path):
        """
        Loads model weights from a checkpoint file.

        :param file_path: Path to the checkpoint file to load.
        """
        self.load_state_dict(torch.load(file_path))

    @property
    def class_names(self) -> list:
        """
        The ordered list of class label strings provided at construction.
        The index of each name corresponds to the integer label the model outputs.
        """
        return self._class_names
