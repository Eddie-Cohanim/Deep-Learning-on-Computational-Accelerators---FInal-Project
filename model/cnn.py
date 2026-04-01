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


_SUPPORTED_POOLING_TYPES = {
    "max": nn.MaxPool2d,
    "avg": nn.AvgPool2d,
}

_SUPPORTED_ACTIVATIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
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
        self._optimizer_class = optimizer_class
        self._num_dataloader_workers = num_dataloader_workers

        _, image_height, image_width = in_size
        self._image_transforms = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std),
        ])

        self._feature_extractor = self._build_feature_extractor()
        self._classifier_head = self._build_classifier_head()
        self._optimizer = optimizer_class(self.parameters(), lr=learning_rate)
        self._loss_function = loss_function

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

    def train_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        """
        Trains the model on images loaded from the given folder.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys 'train_loss' and 'train_accuracy',
            each a list of floats with one value per epoch.
        """
        train_loader = self._load_dataset(dataset_path, shuffle=True)
        self.train()
        epoch_losses = []
        epoch_accuracies = []

        for epoch_index, _ in enumerate(range(self._num_epochs), start=1):
            cumulative_loss = 0.0
            total_correct_predictions = 0
            total_samples = 0

            for images, labels in train_loader:
                self._optimizer.zero_grad()
                class_scores = self.forward(images)
                batch_loss = self._loss_function(class_scores, labels)
                batch_loss.backward()
                self._optimizer.step()

                cumulative_loss += batch_loss.item() * images.shape[0]
                predicted_labels = class_scores.argmax(dim=1)
                total_correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += images.shape[0]

            epoch_average_loss = cumulative_loss / total_samples
            epoch_accuracy = total_correct_predictions / total_samples
            epoch_losses.append(epoch_average_loss)
            epoch_accuracies.append(epoch_accuracy)

            print(f"  Epoch [{epoch_index}/{self._num_epochs}]  loss: {epoch_average_loss:.4f}  accuracy: {epoch_accuracy * 100:.2f}%")

        return {"train_loss": epoch_losses, "train_accuracy": epoch_accuracies}

    def validate_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        """
        Evaluates the model on images loaded from the given folder, without updating weights.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys 'val_loss' and 'val_accuracy' as floats.
        """
        val_loader = self._load_dataset(dataset_path, shuffle=False)
        self.eval()
        cumulative_loss = 0.0
        total_correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
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
        Evaluates the model on images loaded from the given folder and returns predictions.

        Expects the folder structure: dataset_path/<class_name>/<image_file>

        :param dataset_path: Path to the folder containing class subfolders.
        :return: Dictionary with keys 'test_accuracy' (float) and
            'predictions' (1D Tensor of predicted class indices).
        """
        test_loader = self._load_dataset(dataset_path, shuffle=False)
        self.eval()
        total_correct_predictions = 0
        total_samples = 0
        all_predicted_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                class_scores = self.forward(images)
                predicted_labels = class_scores.argmax(dim=1)
                all_predicted_labels.append(predicted_labels)
                total_correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += images.shape[0]

        return {
            "test_accuracy": total_correct_predictions / total_samples,
            "predictions": torch.cat(all_predicted_labels),
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
