import pathlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import kornia.augmentation as K
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
    # Softmax excluded: normalizing hidden activations destroys magnitude info and kills gradients.
}


class CNN(nn.Module):
    """
    Architecture:
        [(Conv2d -> BatchNorm2d -> Activation) * pool_every -> MaxPool2d] * groups
        -> Flatten
        -> [Linear -> Activation -> Dropout] * len(hidden_dims)
        -> Linear(num_classes)
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
        augmentations_config: dict,
    ):
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
        self._augmentations_config = augmentations_config

        _, h, w = in_size
        self._image_transforms = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std),
        ])

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._feature_extractor = self._build_feature_extractor()
        self._classifier_head = self._build_classifier_head()
        self.to(self._device)
        self._optimizer = optimizer_class(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._loss_function = loss_function.to(self._device)

        self._gpu_augmentation_pipeline = self._build_augmentation_pipeline()
        self._gradient_scaler = torch.cuda.amp.GradScaler(enabled=self._device.type == "cuda")

    def _build_augmentation_pipeline(self):
        if not self._augmentations_config.get("enabled", True):
            return None

        config = self._augmentations_config
        augs = []

        if config.get("horizontal_flip", {}).get("enabled", True):
            augs.append(K.RandomHorizontalFlip(p=config["horizontal_flip"]["p"]))

        if config.get("vertical_flip", {}).get("enabled", False):
            augs.append(K.RandomVerticalFlip(p=config["vertical_flip"]["p"]))

        if config.get("rotation", {}).get("enabled", True):
            augs.append(K.RandomRotation(degrees=config["rotation"]["max_degrees"], p=config["rotation"]["p"]))

        if config.get("color_jitter", {}).get("enabled", True):
            jitter = config["color_jitter"]
            augs.append(K.ColorJitter(brightness=jitter["brightness"], contrast=jitter["contrast"], saturation=jitter["saturation"], p=jitter["p"]))

        if config.get("gaussian_blur", {}).get("enabled", True):
            blur = config["gaussian_blur"]
            augs.append(K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(blur["sigma_min"], blur["sigma_max"]), p=blur["p"]))

        if config.get("perspective", {}).get("enabled", True):
            augs.append(K.RandomPerspective(distortion_scale=config["perspective"]["distortion_scale"], p=config["perspective"]["p"]))

        if config.get("gamma", {}).get("enabled", True):
            gamma = config["gamma"]
            augs.append(K.RandomGamma(gamma=(gamma["gamma_min"], gamma["gamma_max"]), gain=(1.0, 1.0), p=gamma["p"]))

        return K.AugmentationSequential(*augs, data_keys=["input"]).to(self._device) if augs else None

    def augmentation_description(self) -> dict:
        if self._gpu_augmentation_pipeline is None:
            return {"enabled": False}
        return self._augmentations_config

    def _load_dataset(self, dataset_path: pathlib.Path, shuffle: bool) -> DataLoader:
        dataset = ImageFolder(root=str(dataset_path), transform=self._image_transforms)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_dataloader_workers > 0,
        )

    def _build_feature_extractor(self) -> nn.Sequential:
        act = _SUPPORTED_ACTIVATIONS[self._activation]
        pool = _SUPPORTED_POOLING_TYPES[self._pooling_type]
        padding = self._conv_kernel_size // 2
        in_ch = self._in_size[0]
        layers = []

        for i, out_ch in enumerate(self._channels, start=1):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=self._conv_kernel_size, padding=padding))
            if self._use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act())
            in_ch = out_ch
            if i % self._pool_every == 0:
                layers.append(pool(kernel_size=self._pool_kernel_size))

        return nn.Sequential(*layers)

    def _compute_feature_output_size(self) -> int:
        x = torch.zeros(1, *self._in_size)
        return self._feature_extractor(x).numel()

    def _build_classifier_head(self) -> nn.Sequential:
        act = _SUPPORTED_ACTIVATIONS[self._activation]
        in_dim = self._compute_feature_output_size()
        layers = []

        for dim in self._hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(act())
            if self._dropout_probability > 0.0:
                layers.append(nn.Dropout(p=self._dropout_probability))
            in_dim = dim

        layers.append(nn.Linear(in_dim, self._num_classes))
        return nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        features = self._feature_extractor(images)
        return self._classifier_head(features.reshape(images.shape[0], -1))

    def train_on_dataset(
        self,
        dataset_path: pathlib.Path,
        val_dataset_path: pathlib.Path = None,
        best_checkpoint_path: pathlib.Path = None,
    ) -> dict:
        train_loader = self._load_dataset(dataset_path, shuffle=True)
        val_loader = self._load_dataset(val_dataset_path, shuffle=False) if val_dataset_path is not None else None
        return self.train_on_data_loaders(train_loader, val_loader, best_checkpoint_path=best_checkpoint_path)

    def train_on_data_loaders(
        self,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader = None,
        best_checkpoint_path: pathlib.Path = None,
        num_epochs_override: int = None,
    ) -> dict:
        train_losses, train_accs = [], []
        use_early_stopping = self._early_stopping_patience > 0 and val_data_loader is not None
        stale_epochs = 0
        best_loss = float("inf")
        best_state = None

        n_epochs = num_epochs_override if num_epochs_override is not None else self._num_epochs
        for epoch in range(1, n_epochs + 1):
            total_loss = 0.0
            n_correct = 0
            n_samples = 0

            self.train()
            for imgs, labels in train_data_loader:
                imgs = imgs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)

                if self._gpu_augmentation_pipeline is not None:
                    imgs = self._gpu_augmentation_pipeline(imgs)

                self._optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self._device.type, enabled=self._device.type == "cuda"):
                    logits = self.forward(imgs)
                    loss = self._loss_function(logits, labels)

                self._gradient_scaler.scale(loss).backward()
                self._gradient_scaler.step(self._optimizer)
                self._gradient_scaler.update()

                total_loss += loss.item() * imgs.shape[0]
                preds = logits.argmax(dim=1)
                n_correct += (preds == labels).sum().item()
                n_samples += imgs.shape[0]

            avg_loss = total_loss / n_samples
            acc = n_correct / n_samples
            train_losses.append(avg_loss)
            train_accs.append(acc)

            print(f"  Epoch [{epoch}/{n_epochs}]  loss: {avg_loss:.4f}  accuracy: {acc * 100:.2f}%", end="", flush=True)

            if use_early_stopping:
                val = self.validate_on_data_loader(val_data_loader)
                val_loss, val_acc = val["val_loss"], val["val_accuracy"]
                print(f"  val_loss: {val_loss:.4f}  val_accuracy: {val_acc * 100:.2f}%", end="", flush=True)

                if val_loss < best_loss:
                    best_loss = val_loss
                    stale_epochs = 0
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                    if best_checkpoint_path is not None:
                        torch.save(best_state, best_checkpoint_path)
                else:
                    stale_epochs += 1

                if stale_epochs >= self._early_stopping_patience:
                    print(f"\n  Early stopping triggered after {epoch} epochs (no improvement for {self._early_stopping_patience} epochs).")
                    self.load_state_dict(best_state)
                    break

            print()

        return {
            "train_loss": train_losses,
            "train_accuracy": train_accs,
            "epochs_trained": len(train_losses),
        }

    def validate_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        val_loader = self._load_dataset(dataset_path, shuffle=False)
        return self.validate_on_data_loader(val_loader)

    def validate_on_data_loader(self, val_data_loader: DataLoader) -> dict:
        self.eval()
        total_loss = 0.0
        n_correct = 0
        n_samples = 0

        with torch.no_grad():
            for imgs, labels in val_data_loader:
                imgs = imgs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                logits = self.forward(imgs)
                loss = self._loss_function(logits, labels)
                total_loss += loss.item() * imgs.shape[0]
                preds = logits.argmax(dim=1)
                n_correct += (preds == labels).sum().item()
                n_samples += imgs.shape[0]

        return {
            "val_loss": total_loss / n_samples,
            "val_accuracy": n_correct / n_samples,
        }

    def test_on_dataset(self, dataset_path: pathlib.Path) -> dict:
        dataset = ImageFolder(root=str(dataset_path), transform=self._image_transforms)
        test_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_dataloader_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=self._num_dataloader_workers > 0,
        )
        class_names = dataset.classes

        self.eval()
        n_correct = 0
        n_samples = 0
        pred_batches, true_batches, conf_batches = [], [], []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                logits = self.forward(imgs)
                probs = torch.softmax(logits, dim=1)
                batch_preds = logits.argmax(dim=1)
                pred_batches.append(batch_preds.cpu())
                true_batches.append(labels.cpu())
                conf_batches.append(probs.max(dim=1).values.cpu())
                n_correct += (batch_preds == labels).sum().item()
                n_samples += imgs.shape[0]

        preds = torch.cat(pred_batches)
        targets = torch.cat(true_batches)
        confidences = torch.cat(conf_batches)

        per_class_results = {}
        for idx, name in enumerate(class_names):
            tp = ((preds == idx) & (targets == idx)).sum().item()
            fp = ((preds == idx) & (targets != idx)).sum().item()
            fn = ((preds != idx) & (targets == idx)).sum().item()
            tn = n_samples - tp - fp - fn
            n_class = tp + fn

            pred_mask = preds == idx
            avg_conf = confidences[pred_mask].mean().item() if pred_mask.sum().item() > 0 else 0.0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_class if n_class > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_results[name] = {
                "total_samples_tested": n_class,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "mean_confidence": round(avg_conf, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        raw_predictions = [
            {
                "ground_truth": class_names[t],
                "predicted": class_names[p],
                "confidence": round(c, 4),
            }
            for t, p, c in zip(targets.tolist(), preds.tolist(), confidences.tolist())
        ]

        return {
            "test_accuracy": n_correct / n_samples,
            "per_class_results": per_class_results,
            "raw_predictions": raw_predictions,
        }

    def save_checkpoint(self, file_path: pathlib.Path):
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path: pathlib.Path):
        self.load_state_dict(torch.load(file_path))

    @property
    def class_names(self) -> list:
        return self._class_names
