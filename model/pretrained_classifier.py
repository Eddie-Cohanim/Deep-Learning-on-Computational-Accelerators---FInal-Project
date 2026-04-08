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


_SUPPORTED_MODEL_TYPES = ("dino", "efficientnet")

_DINO_EMBED_DIM = 384  # ViT-S/16 CLS token output dimensionality


class _DinoWithLinearHead(nn.Module):
    """
    Wraps a DINO ViT backbone and a single linear classification head
    into one nn.Module so that both are captured by state_dict().
    """

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        """
        :param backbone: A pretrained DINO ViT backbone that outputs a
            CLS token embedding of shape (batch_size, embed_dim).
        :param num_classes: Number of target output classes.
        """
        super().__init__()
        self._backbone = backbone
        self._classification_head = nn.Linear(_DINO_EMBED_DIM, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """
        :param images: Input batch of shape (batch_size, channels, height, width).
        :return: Raw class score tensor of shape (batch_size, num_classes).
        """
        cls_token_embedding = self._backbone(images)
        return self._classification_head(cls_token_embedding)


class PretrainedClassifier(nn.Module):
    """
    Wraps a pretrained backbone (DINO ViT-S/16 or EfficientNet-B0) with a
    new classification head for fine-tuning on a custom image dataset.

    The backbone weights can be frozen (linear probe) or fully fine-tuned,
    controlled by the fine_tune_backbone flag.

    The public interface — train_on_dataset, validate_on_dataset,
    test_on_dataset, save_checkpoint — is identical to the CNN class so
    that the two are interchangeable in main.py.
    """

    def __init__(
        self,
        model_type: str,
        class_names: list,
        fine_tune_backbone: bool,
        image_normalization_mean: list,
        image_normalization_std: list,
        num_epochs: int,
        optimizer_class: type,
        loss_function: nn.Module,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        num_dataloader_workers: int,
        early_stopping_patience: int,
        run_augmentation: bool,
        number_of_augmented_copies_per_image: int,
        augmentation_rotation_max_degrees: float,
        augmentation_brightness_jitter: float,
        augmentation_contrast_jitter: float,
        augmentation_saturation_jitter: float,
    ) -> None:
        """
        :param model_type: Which pretrained backbone to load.
            Accepted values: 'dino', 'efficientnet'.
        :param class_names: Ordered list of class label strings. The index of
            each name corresponds to the integer label the model outputs.
        :param fine_tune_backbone: When False, the backbone weights are frozen
            and only the new classification head is trained (linear probe).
            When True, the entire network is trained end-to-end.
        :param image_normalization_mean: Per-channel mean used to normalize
            input images. Should be the ImageNet mean [0.485, 0.456, 0.406]
            to match the statistics used when pretraining the backbone.
        :param image_normalization_std: Per-channel std used to normalize
            input images. Should be the ImageNet std [0.229, 0.224, 0.225].
        :param num_epochs: Maximum number of full passes over the training set.
        :param optimizer_class: The optimizer class to instantiate, e.g.
            torch.optim.Adam or torch.optim.AdamW.
        :param loss_function: An instantiated loss function, e.g.
            nn.CrossEntropyLoss().
        :param learning_rate: Learning rate passed to the optimizer.
        :param weight_decay: L2 regularization coefficient passed to the
            optimizer.
        :param batch_size: Number of images per mini-batch.
        :param num_dataloader_workers: Number of worker processes for the
            DataLoader. 0 loads images in the main process.
        :param early_stopping_patience: Number of consecutive epochs without
            improvement in validation loss before training stops early.
            Set to 0 to disable early stopping.
        :param run_augmentation: Whether to apply online augmentation during
            training. When False, images are loaded as-is.
        :param number_of_augmented_copies_per_image: Number of additional
            augmented versions generated per training image. The effective
            dataset size is (1 + this value) times the original size.
        :param augmentation_rotation_max_degrees: Maximum rotation in degrees
            for the random rotation augmentation.
        :param augmentation_brightness_jitter: Maximum fractional brightness
            change for the color jitter augmentation.
        :param augmentation_contrast_jitter: Maximum fractional contrast
            change for the color jitter augmentation.
        :param augmentation_saturation_jitter: Maximum fractional saturation
            change for the color jitter augmentation.
        """
        super().__init__()

        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {list(_SUPPORTED_MODEL_TYPES)}"
            )

        self._model_type = model_type
        self._class_names = list(class_names)
        self._num_classes = len(class_names)
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_dataloader_workers = num_dataloader_workers
        self._early_stopping_patience = early_stopping_patience
        self._run_augmentation = run_augmentation
        self._number_of_augmented_copies_per_image = number_of_augmented_copies_per_image

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

        # Both DINO and EfficientNet were pretrained on 224×224 images with
        # ImageNet normalization, so these transforms are correct for both.
        self._image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_normalization_mean,
                std=image_normalization_std,
            ),
        ])

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._network = self._build_network(model_type, fine_tune_backbone)
        self.to(self._device)

        # Only pass parameters that require gradients to the optimizer so that
        # frozen backbone weights do not waste optimizer state memory.
        trainable_parameters = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        ]
        self._optimizer = optimizer_class(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self._loss_function = loss_function
        self._grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self._device.type == "cuda"
        )

    def _build_network(self, model_type: str, fine_tune_backbone: bool) -> nn.Module:
        """
        Loads the pretrained backbone, optionally freezes its weights, and
        attaches a new linear classification head sized for the target classes.

        :param model_type: 'dino' or 'efficientnet'.
        :param fine_tune_backbone: Whether to keep backbone weights trainable.
        :return: A single nn.Module ready for forward passes and training.
        """
        if model_type == "dino":
            return self._build_dino_network(fine_tune_backbone)
        else:
            return self._build_efficientnet_network(fine_tune_backbone)

    def _build_dino_network(self, fine_tune_backbone: bool) -> nn.Module:
        """
        Loads DINO ViT-S/16 from torch.hub and wraps it with a linear head.

        The backbone outputs a 384-dimensional CLS token embedding.
        The linear head maps that embedding to num_classes logits.

        :param fine_tune_backbone: When False, all backbone parameters are
            frozen and only the linear head remains trainable.
        :return: A _DinoWithLinearHead module.
        """
        backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vits16",
        )

        if not fine_tune_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad = False

        return _DinoWithLinearHead(
            backbone=backbone,
            num_classes=self._num_classes,
        )

    def _build_efficientnet_network(self, fine_tune_backbone: bool) -> nn.Module:
        """
        Loads EfficientNet-B0 with ImageNet weights and replaces the final
        fully-connected layer with one sized for num_classes.

        EfficientNet's classifier is nn.Sequential(Dropout(p=0.2), Linear(1280, 1000)).
        Only the final Linear is replaced; the Dropout layer is preserved.
        The replacement layer is always trainable regardless of fine_tune_backbone.

        :param fine_tune_backbone: When False, all parameters except the new
            classification layer are frozen.
        :return: The modified EfficientNet-B0 nn.Module.
        """
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        network = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        if not fine_tune_backbone:
            for parameter in network.parameters():
                parameter.requires_grad = False

        original_in_features = network.classifier[1].in_features
        network.classifier[1] = nn.Linear(original_in_features, self._num_classes)
        # The newly created layer always has requires_grad=True by default,
        # so it remains trainable even when the backbone is frozen.

        return network

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
        sequence before the base transform is applied.

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

    def forward(self, images: Tensor) -> Tensor:
        """
        Runs a forward pass through the pretrained backbone and classification head.

        :param images: Input batch of shape (batch_size, channels, height, width).
        :return: Raw class score tensor of shape (batch_size, num_classes).
        """
        return self._network(images)

    def train_on_dataset(
        self,
        dataset_path: pathlib.Path,
        val_dataset_path: pathlib.Path = None,
    ) -> dict:
        """
        Trains the model on images loaded from the given folder.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        If val_dataset_path is provided and early_stopping_patience > 0,
        training stops early when validation loss has not improved for that
        many epochs. The best weights are restored at the end.

        :param dataset_path: Path to the training folder.
        :param val_dataset_path: Optional path to the validation split.
        :return: Dictionary with keys 'train_loss' and 'train_accuracy',
            each a list of floats with one value per epoch.
        """
        if self._run_augmentation:
            train_loader = self._load_augmented_training_dataset(dataset_path)
        else:
            train_loader = self._load_dataset(dataset_path, shuffle=True)

        val_loader = (
            self._load_dataset(val_dataset_path, shuffle=False)
            if val_dataset_path is not None
            else None
        )
        return self.train_on_data_loaders(train_loader, val_loader)

    def train_on_data_loaders(
        self,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader = None,
    ) -> dict:
        """
        Trains the model using pre-built DataLoaders.

        train_on_dataset delegates to this method after constructing its
        DataLoaders from disk paths.

        :param train_data_loader: DataLoader over the training set.
        :param val_data_loader: Optional DataLoader over the validation set,
            used for early stopping.
        :return: Dictionary with keys 'train_loss' and 'train_accuracy',
            each a list of floats with one value per epoch.
        """
        epoch_losses = []
        epoch_accuracies = []

        early_stopping_enabled = (
            self._early_stopping_patience > 0 and val_data_loader is not None
        )
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
                with torch.amp.autocast(
                    device_type=self._device.type,
                    enabled=self._device.type == "cuda",
                ):
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

            print(
                f"  Epoch [{epoch_index}/{self._num_epochs}]"
                f"  loss: {epoch_average_loss:.4f}"
                f"  accuracy: {epoch_accuracy * 100:.2f}%",
                end="",
                flush=True,
            )

            if early_stopping_enabled:
                val_results = self.validate_on_data_loader(val_data_loader)
                current_val_loss = val_results["val_loss"]
                print(f"  val_loss: {current_val_loss:.4f}", end="", flush=True)

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_without_improvement = 0
                    best_weights = {
                        key: value.cpu().clone()
                        for key, value in self.state_dict().items()
                    }
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self._early_stopping_patience:
                    print(
                        f"\n  Early stopping triggered after {epoch_index} epochs"
                        f" (no improvement for {self._early_stopping_patience} epochs)."
                    )
                    self.load_state_dict({
                        key: value.to(self._device)
                        for key, value in best_weights.items()
                    })
                    break

            print()

        return {"train_loss": epoch_losses, "train_accuracy": epoch_accuracies}

    def validate_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        """
        Evaluates the model on images loaded from the given folder, without
        updating weights.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys 'val_loss' and 'val_accuracy' as floats.
        """
        val_loader = self._load_dataset(dataset_path, shuffle=False)
        return self.validate_on_data_loader(val_loader)

    def validate_on_data_loader(self, val_data_loader: DataLoader) -> dict:
        """
        Evaluates the model using a pre-built DataLoader, without updating weights.

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
        Evaluates the model on the test set and returns per-class precision,
        recall, F1, sample counts, and raw predictions.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys:
            - 'test_accuracy': overall accuracy as a float
            - 'per_class_results': dict mapping each class name to its metrics
            - 'raw_predictions': list of per-sample prediction records
        """
        dataset = ImageFolder(
            root=str(dataset_path),
            transform=self._image_transforms,
        )
        test_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_dataloader_workers > 0,
        )
        class_names_from_dataset = dataset.classes

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
            true_positives = (
                (all_predicted_labels_tensor == class_index) &
                (all_true_labels_tensor == class_index)
            ).sum().item()
            false_positives = (
                (all_predicted_labels_tensor == class_index) &
                (all_true_labels_tensor != class_index)
            ).sum().item()
            false_negatives = (
                (all_predicted_labels_tensor != class_index) &
                (all_true_labels_tensor == class_index)
            ).sum().item()
            true_negatives = (
                total_samples - true_positives - false_positives - false_negatives
            )
            total_class_samples = true_positives + false_negatives

            predicted_as_this_class_mask = all_predicted_labels_tensor == class_index
            if predicted_as_this_class_mask.sum().item() > 0:
                mean_confidence = all_confidence_scores_tensor[
                    predicted_as_this_class_mask
                ].mean().item()
            else:
                mean_confidence = 0.0

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0 else 0.0
            )
            recall = (
                true_positives / total_class_samples
                if total_class_samples > 0 else 0.0
            )
            f1_score = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

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

    def save_checkpoint(self, file_path: pathlib.Path) -> None:
        """
        Saves the current model weights to a file.

        :param file_path: Destination path for the checkpoint file.
        """
        torch.save(self.state_dict(), file_path)

    @property
    def class_names(self) -> list:
        """
        The ordered list of class label strings provided at construction.
        """
        return self._class_names
