import pathlib
from typing import Callable, List, Optional, Sequence, Set

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import kornia.augmentation as K
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# Maps backbone name → (torchvision loader name, family identifier)
# Family determines where the classification head lives and how to replace it.
_BACKBONE_REGISTRY = {
    # ResNet / Wide-ResNet: head is backbone.fc (single Linear)
    "resnet18":          "resnet",
    "resnet34":          "resnet",
    "resnet50":          "resnet",
    "resnet101":         "resnet",
    "resnet152":         "resnet",
    "wide_resnet50_2":   "resnet",
    "wide_resnet101_2":  "resnet",
    # EfficientNet: head is backbone.classifier (Sequential ending in Linear)
    "efficientnet_b0":   "efficientnet",
    "efficientnet_b1":   "efficientnet",
    "efficientnet_b2":   "efficientnet",
    "efficientnet_b3":   "efficientnet",
    "efficientnet_b4":   "efficientnet",
    "efficientnet_b5":   "efficientnet",
    "efficientnet_b6":   "efficientnet",
    "efficientnet_b7":   "efficientnet",
    # MobileNet: head is backbone.classifier (Sequential ending in Linear)
    "mobilenet_v2":          "mobilenet",
    "mobilenet_v3_small":    "mobilenet",
    "mobilenet_v3_large":    "mobilenet",
    # DenseNet: head is backbone.classifier (single Linear)
    "densenet121":  "densenet",
    "densenet161":  "densenet",
    "densenet169":  "densenet",
    "densenet201":  "densenet",
    # VGG: head is backbone.classifier (Sequential); we replace only the final Linear layer
    "vgg11":     "vgg",
    "vgg11_bn":  "vgg",
    "vgg13":     "vgg",
    "vgg13_bn":  "vgg",
    "vgg16":     "vgg",
    "vgg16_bn":  "vgg",
    "vgg19":     "vgg",
    "vgg19_bn":  "vgg",
}


def _get_backbone_feature_dim(backbone: nn.Module, family: str) -> int:
    """
    Returns the number of input features expected by the existing classification head.
    This is the dimension the new head must accept.
    """
    if family == "resnet":
        return backbone.fc.in_features
    if family in ("efficientnet", "mobilenet"):
        return backbone.classifier[-1].in_features
    if family == "densenet":
        return backbone.classifier.in_features
    if family == "vgg":
        return backbone.classifier[-1].in_features
    raise ValueError(f"Unrecognised backbone family: {family!r}")


def _attach_classification_head(
    backbone: nn.Module,
    family: str,
    new_head: nn.Module,
) -> None:
    """
    Replaces the existing classification head on the backbone with new_head, in-place.
    For VGG only the final linear layer is replaced; all other families replace the
    entire head attribute.
    """
    if family == "resnet":
        backbone.fc = new_head
    elif family in ("efficientnet", "mobilenet", "densenet"):
        backbone.classifier = new_head
    elif family == "vgg":
        # backbone.classifier is a Sequential; keep the pretrained FC layers,
        # only swap the final Linear so the feature mapping is preserved.
        backbone.classifier[-1] = new_head
    else:
        raise ValueError(f"Unrecognised backbone family: {family!r}")


def _build_classification_head(
    in_features: int,
    hidden_dims: Sequence[int],
    dropout_probability: float,
    num_classes: int,
) -> nn.Module:
    """
    Builds a fully-connected classification head.

    If hidden_dims is empty, returns a single Linear(in_features, num_classes).
    Otherwise, inserts [Linear → ReLU → Dropout] blocks for each hidden dim,
    followed by a final Linear(last_dim, num_classes).

    :param in_features: Input feature dimension (output of the backbone).
    :param hidden_dims: Sizes of intermediate hidden layers (may be empty).
    :param dropout_probability: Dropout rate applied after each hidden layer.
    :param num_classes: Number of output classes.
    :return: An nn.Module representing the classification head.
    """
    if not hidden_dims:
        return nn.Linear(in_features, num_classes)

    layers = []
    current_dim = in_features
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_probability > 0.0:
            layers.append(nn.Dropout(p=dropout_probability))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)


class PretrainedModel(nn.Module):
    """
    A transfer-learning wrapper around any supported torchvision pretrained backbone.

    The pretrained feature extractor is loaded with ImageNet weights. Its classification
    head is replaced with a new head sized for the target classes. The backbone can
    optionally be frozen during early training and unfrozen later with a smaller learning
    rate (discriminative fine-tuning).

    Architecture:
        torchvision backbone (pretrained, optionally frozen)
        → replaced head: [Linear → ReLU → Dropout] * len(hidden_dims) → Linear(num_classes)

    Supported backbones: resnet18/34/50/101/152, wide_resnet50_2/101_2,
        efficientnet_b0–b7, mobilenet_v2/v3_small/v3_large,
        densenet121/161/169/201, vgg11/13/16/19 (with/without bn).

    The training, validation, and test interface is identical to CNN so that either
    model can be passed to CrossValidator without changes.
    """

    def __init__(
        self,
        backbone_name: str,
        class_names: list,
        hidden_dims: Sequence[int],
        dropout_probability: float,
        freeze_backbone: bool,
        unfreeze_after_epoch: int,
        learning_rate: float,
        backbone_learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        batch_size: int,
        num_dataloader_workers: int,
        optimizer_class: type,
        loss_function: nn.Module,
        image_normalization_mean: list,
        image_normalization_std: list,
        in_size: tuple,
        early_stopping_patience: int,
        augmentations_config: dict,
    ):
        """
        :param backbone_name: Name of the torchvision backbone, e.g. 'resnet50'.
        :param class_names: Ordered list of class label strings.
        :param hidden_dims: Hidden layer sizes for the new classification head.
            Pass an empty list to use a single linear layer.
        :param dropout_probability: Dropout rate in the new head. Set to 0.0 to disable.
        :param freeze_backbone: If True, all backbone parameters are frozen at startup.
            Only the new classification head is trained until unfreeze_after_epoch.
        :param unfreeze_after_epoch: Epoch at which the backbone is unfrozen. Only
            used when freeze_backbone is True. Set to 0 to keep the backbone frozen
            for the entire training run.
        :param learning_rate: Learning rate for the classification head (and for the
            backbone when not frozen or when freeze_backbone is False).
        :param backbone_learning_rate: Learning rate applied to backbone parameters
            after they are unfrozen. Should be much smaller than learning_rate.
        :param weight_decay: L2 regularisation coefficient for the optimizer.
        :param num_epochs: Maximum number of training epochs.
        :param batch_size: Number of images per batch.
        :param num_dataloader_workers: Number of DataLoader worker processes.
        :param optimizer_class: Optimizer class, e.g. torch.optim.AdamW.
        :param loss_function: Instantiated loss function, e.g. nn.CrossEntropyLoss().
        :param image_normalization_mean: Per-channel mean for input normalisation.
        :param image_normalization_std: Per-channel std for input normalisation.
        :param in_size: Input image shape as (channels, height, width). Images are
            resized to (height, width) before being fed to the backbone.
        :param early_stopping_patience: Number of epochs without val-loss improvement
            before stopping early. Set to 0 to disable.
        :param augmentations_config: The "augmentations" block from config.json.
        """
        super().__init__()

        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Supported: {sorted(_BACKBONE_REGISTRY.keys())}"
            )

        self._backbone_name = backbone_name
        self._backbone_family = _BACKBONE_REGISTRY[backbone_name]
        self._class_names = list(class_names)
        self._num_classes = len(class_names)
        self._freeze_backbone = freeze_backbone
        self._unfreeze_after_epoch = unfreeze_after_epoch
        self._learning_rate = learning_rate
        self._backbone_learning_rate = backbone_learning_rate
        self._weight_decay = weight_decay
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._num_dataloader_workers = num_dataloader_workers
        self._optimizer_class = optimizer_class
        self._early_stopping_patience = early_stopping_patience
        self._augmentations_config = augmentations_config
        self._in_size = in_size

        _, image_height, image_width = in_size
        self._image_transforms = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std),
        ])

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained backbone
        torchvision_loader = getattr(models, backbone_name)
        pretrained_backbone = torchvision_loader(pretrained=True)

        # Build and attach the new classification head
        feature_dim = _get_backbone_feature_dim(pretrained_backbone, self._backbone_family)
        new_head = _build_classification_head(
            in_features=feature_dim,
            hidden_dims=hidden_dims,
            dropout_probability=dropout_probability,
            num_classes=self._num_classes,
        )

        # Record head parameter ids BEFORE moving to device so we can identify them.
        # nn.Parameter objects are stable Python objects — their id() does not change
        # when .to(device) is called (only their internal .data changes).
        self._head_param_ids: Set[int] = {id(param) for param in new_head.parameters()}

        _attach_classification_head(pretrained_backbone, self._backbone_family, new_head)

        # Register backbone as a tracked submodule (the only one — avoids duplicate params)
        self._backbone = pretrained_backbone
        self.to(self._device)

        # Freeze all backbone params except the new head
        if freeze_backbone:
            for param in self._backbone.parameters():
                if id(param) not in self._head_param_ids:
                    param.requires_grad = False

        self._loss_function = loss_function.to(self._device)

        trainable_params = [param for param in self._backbone.parameters() if param.requires_grad]
        self._optimizer = optimizer_class(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self._gpu_augmentation_pipeline = self._build_augmentation_pipeline()
        self._gradient_scaler = torch.cuda.amp.GradScaler(enabled=self._device.type == "cuda")

    def _build_augmentation_pipeline(self):
        """
        Constructs the Kornia GPU augmentation pipeline from the augmentations config.

        Returns None if augmentation is disabled globally or no transforms are enabled.
        """
        if not self._augmentations_config.get("enabled", True):
            return None

        active_transforms = []

        horizontal_flip_config = self._augmentations_config.get("horizontal_flip", {})
        if horizontal_flip_config.get("enabled", True):
            active_transforms.append(K.RandomHorizontalFlip(p=horizontal_flip_config["p"]))

        vertical_flip_config = self._augmentations_config.get("vertical_flip", {})
        if vertical_flip_config.get("enabled", False):
            active_transforms.append(K.RandomVerticalFlip(p=vertical_flip_config["p"]))

        rotation_config = self._augmentations_config.get("rotation", {})
        if rotation_config.get("enabled", True):
            active_transforms.append(
                K.RandomRotation(degrees=rotation_config["max_degrees"], p=rotation_config["p"])
            )

        color_jitter_config = self._augmentations_config.get("color_jitter", {})
        if color_jitter_config.get("enabled", True):
            active_transforms.append(
                K.ColorJitter(
                    brightness=color_jitter_config["brightness"],
                    contrast=color_jitter_config["contrast"],
                    saturation=color_jitter_config["saturation"],
                    p=color_jitter_config["p"],
                )
            )

        gaussian_blur_config = self._augmentations_config.get("gaussian_blur", {})
        if gaussian_blur_config.get("enabled", True):
            active_transforms.append(
                K.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(gaussian_blur_config["sigma_min"], gaussian_blur_config["sigma_max"]),
                    p=gaussian_blur_config["p"],
                )
            )

        perspective_config = self._augmentations_config.get("perspective", {})
        if perspective_config.get("enabled", True):
            active_transforms.append(
                K.RandomPerspective(
                    distortion_scale=perspective_config["distortion_scale"],
                    p=perspective_config["p"],
                )
            )

        gamma_config = self._augmentations_config.get("gamma", {})
        if gamma_config.get("enabled", True):
            active_transforms.append(
                K.RandomGamma(
                    gamma=(gamma_config["gamma_min"], gamma_config["gamma_max"]),
                    gain=(1.0, 1.0),
                    p=gamma_config["p"],
                )
            )

        if not active_transforms:
            return None

        return K.AugmentationSequential(*active_transforms, data_keys=["input"]).to(self._device)

    def augmentation_description(self) -> dict:
        """
        Returns the augmentation config as stored in config.json.
        Written into results.json for each experiment.
        """
        if self._gpu_augmentation_pipeline is None:
            return {"enabled": False}
        return self._augmentations_config

    def forward(self, images: Tensor) -> Tensor:
        """
        Runs a forward pass through the backbone (feature extractor + classification head).

        :param images: Input batch of shape (batch_size, channels, height, width).
        :return: Raw class score tensor of shape (batch_size, num_classes).
        """
        return self._backbone(images)

    def _load_dataset(self, dataset_path: pathlib.Path, shuffle: bool) -> DataLoader:
        """
        Loads images from a folder with the structure:
            dataset_path/<class_name>/<image_file>

        :param dataset_path: Root folder containing one subfolder per class.
        :param shuffle: Whether to shuffle images on each iteration.
        :return: A DataLoader over the images in the given folder.
        """
        dataset = ImageFolder(root=str(dataset_path), transform=self._image_transforms)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
        )

    def _unfreeze_backbone_and_rebuild_optimizer(self) -> None:
        """
        Unfreezes all backbone parameters and rebuilds the optimizer with
        discriminative learning rates: backbone_learning_rate for pretrained
        layers, learning_rate for the new classification head.
        """
        for param in self._backbone.parameters():
            param.requires_grad = True

        head_params = [
            param for param in self._backbone.parameters()
            if id(param) in self._head_param_ids
        ]
        backbone_params = [
            param for param in self._backbone.parameters()
            if id(param) not in self._head_param_ids
        ]

        self._optimizer = self._optimizer_class(
            [
                {"params": backbone_params, "lr": self._backbone_learning_rate},
                {"params": head_params,     "lr": self._learning_rate},
            ],
            weight_decay=self._weight_decay,
        )

    def train_on_dataset(
        self,
        dataset_path: pathlib.Path,
        val_dataset_path: pathlib.Path = None,
    ) -> dict:
        """
        Trains the model on images loaded from the given folder.

        :param dataset_path: Path to training split with class subfolders.
        :param val_dataset_path: Optional validation split path for early stopping.
        :return: Dict with 'train_loss' and 'train_accuracy' lists.
        """
        train_loader = self._load_dataset(dataset_path, shuffle=True)
        val_loader = (
            self._load_dataset(val_dataset_path, shuffle=False)
            if val_dataset_path is not None else None
        )
        return self.train_on_data_loaders(train_loader, val_loader)

    def train_on_data_loaders(
        self,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader = None,
    ) -> dict:
        """
        Trains the model using pre-built DataLoaders.

        When freeze_backbone is True and unfreeze_after_epoch > 0, the backbone is
        automatically unfrozen at the specified epoch with discriminative learning rates.

        :param train_data_loader: DataLoader over the training set.
        :param val_data_loader: Optional DataLoader over the validation set.
        :return: Dict with 'train_loss' and 'train_accuracy' lists (one value per epoch).
        """
        epoch_losses = []
        epoch_accuracies = []

        early_stopping_enabled = self._early_stopping_patience > 0 and val_data_loader is not None
        epochs_without_improvement = 0
        best_val_loss = float("inf")
        best_weights = None
        backbone_has_been_unfrozen = not self._freeze_backbone

        for epoch_index in range(1, self._num_epochs + 1):

            # Unfreeze backbone when the scheduled epoch is reached
            if (
                not backbone_has_been_unfrozen
                and self._unfreeze_after_epoch > 0
                and epoch_index >= self._unfreeze_after_epoch
            ):
                self._unfreeze_backbone_and_rebuild_optimizer()
                backbone_has_been_unfrozen = True
                print(f"  Backbone unfrozen at epoch {epoch_index} with lr={self._backbone_learning_rate}.")

            cumulative_loss = 0.0
            total_correct_predictions = 0
            total_samples = 0

            self.train()
            for images, labels in train_data_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                if self._gpu_augmentation_pipeline is not None:
                    images = self._gpu_augmentation_pipeline(images)

                self._optimizer.zero_grad()

                with torch.autocast(device_type=self._device.type, enabled=self._device.type == "cuda"):
                    class_scores = self.forward(images)
                    batch_loss = self._loss_function(class_scores, labels)

                self._gradient_scaler.scale(batch_loss).backward()
                self._gradient_scaler.step(self._optimizer)
                self._gradient_scaler.update()

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
                    best_weights = {key: value.cpu().clone() for key, value in self.state_dict().items()}
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self._early_stopping_patience:
                    print(
                        f"\n  Early stopping triggered after {epoch_index} epochs"
                        f" (no improvement for {self._early_stopping_patience} epochs)."
                    )
                    self.load_state_dict({key: value.to(self._device) for key, value in best_weights.items()})
                    break

            print()

        return {"train_loss": epoch_losses, "train_accuracy": epoch_accuracies}

    def validate_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        """
        Evaluates the model on images loaded from the given folder.

        :param dataset_path: Path to validation split with class subfolders.
        :return: Dict with 'val_loss' and 'val_accuracy' as floats.
        """
        val_loader = self._load_dataset(dataset_path, shuffle=False)
        return self.validate_on_data_loader(val_loader)

    def validate_on_data_loader(self, val_data_loader: DataLoader) -> dict:
        """
        Evaluates the model using a pre-built DataLoader without updating weights.

        :param val_data_loader: DataLoader over the validation set.
        :return: Dict with 'val_loss' and 'val_accuracy' as floats.
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
        Evaluates the model on the test set and returns per-class metrics.

        :param dataset_path: Path to test split with class subfolders.
        :return: Dict with 'test_accuracy', 'per_class_results', and 'raw_predictions'.
        """
        dataset = ImageFolder(root=str(dataset_path), transform=self._image_transforms)
        test_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
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
                (all_predicted_labels_tensor == class_index) & (all_true_labels_tensor == class_index)
            ).sum().item()
            false_positives = (
                (all_predicted_labels_tensor == class_index) & (all_true_labels_tensor != class_index)
            ).sum().item()
            false_negatives = (
                (all_predicted_labels_tensor != class_index) & (all_true_labels_tensor == class_index)
            ).sum().item()
            true_negatives = total_samples - true_positives - false_positives - false_negatives
            total_class_samples = true_positives + false_negatives

            predicted_as_this_class_mask = all_predicted_labels_tensor == class_index
            if predicted_as_this_class_mask.sum().item() > 0:
                mean_confidence = all_confidence_scores_tensor[predicted_as_this_class_mask].mean().item()
            else:
                mean_confidence = 0.0

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0 else 0.0
            )
            recall = true_positives / total_class_samples if total_class_samples > 0 else 0.0
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
                "ground_truth": class_names_from_dataset[true_label],
                "predicted": class_names_from_dataset[predicted_label],
                "confidence": round(confidence, 4),
            }
            for true_label, predicted_label, confidence in zip(
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
        Saves the full model state (backbone + head weights) to a file.

        :param file_path: Destination path for the checkpoint file.
        """
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path: pathlib.Path) -> None:
        """
        Loads model weights from a checkpoint file.

        :param file_path: Path to the checkpoint file to load.
        """
        self.load_state_dict(torch.load(file_path, map_location=self._device))

    @property
    def class_names(self) -> list:
        """Ordered list of class label strings provided at construction."""
        return self._class_names
