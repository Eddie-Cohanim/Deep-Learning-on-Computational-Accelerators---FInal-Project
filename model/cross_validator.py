import collections
import pathlib
import random
import statistics
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model.augmented_dataset import AugmentedSampleListDataset, SampleListDataset
from model.Augmentations.horizontal_flip_augmentation import HorizontalFlipAugmentation
from model.Augmentations.rotation_augmentation import RotationAugmentation
from model.Augmentations.color_jitter_augmentation import ColorJitterAugmentation
from model.Augmentations.gaussian_blur_augmentation import GaussianBlurAugmentation
from model.cnn import CNN


class CrossValidator:
    """
    Orchestrates stratified K-fold cross-validation for the CNN classifier.

    The combined train and validation splits are divided into K folds.
    For each fold, a freshly initialised CNN is trained on the K-1 remaining
    folds and evaluated on the held-out fold. After all folds are complete,
    a final model is trained on the entire train+val pool and evaluated on
    the held-out test set.

    Results include per-fold metrics and aggregate mean ± standard deviation
    for both validation loss and validation accuracy.
    """

    def __init__(
        self,
        num_folds: int,
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
        augmentation_rotation_max_degrees: float,
        augmentation_brightness_jitter: float,
        augmentation_contrast_jitter: float,
        augmentation_saturation_jitter: float,
    ) -> None:
        """
        :param num_folds: Number of folds K. Must be at least 2 and no greater
            than the number of samples in the smallest class.
        :param in_size: Shape of a single input image as (channels, height, width).
        :param class_names: Ordered list of class label strings.
        :param channels: Output channel count for each convolutional layer.
        :param pool_every: Number of conv layers between each pooling step.
        :param hidden_dims: Output dimension for each fully-connected hidden layer.
        :param num_epochs: Maximum number of training epochs per fold.
        :param optimizer_class: The optimizer class to instantiate, e.g. torch.optim.AdamW.
        :param loss_function: An instantiated loss function, e.g. nn.CrossEntropyLoss().
        :param conv_kernel_size: Kernel size for all convolutional layers.
        :param pooling_type: Either 'max' or 'avg'.
        :param pool_kernel_size: Kernel size for pooling layers.
        :param image_normalization_mean: Per-channel mean for input normalisation.
        :param image_normalization_std: Per-channel std for input normalisation.
        :param num_dataloader_workers: Number of worker processes for loading images.
        :param batch_size: Number of images per batch.
        :param activation: Activation function name, e.g. 'relu'.
        :param use_batchnorm: Whether to apply batch normalisation after each conv layer.
        :param dropout_probability: Dropout probability for fully-connected layers.
        :param learning_rate: Learning rate for the optimizer.
        :param weight_decay: L2 regularisation coefficient.
        :param early_stopping_patience: Epochs without validation improvement before stopping.
        :param number_of_augmented_copies_per_image: Online augmented copies per training image.
        :param augmentation_rotation_max_degrees: Max rotation angle for online augmentation.
        :param augmentation_brightness_jitter: Brightness jitter magnitude.
        :param augmentation_contrast_jitter: Contrast jitter magnitude.
        :param augmentation_saturation_jitter: Saturation jitter magnitude.
        """
        self._num_folds = num_folds
        self._in_size = in_size
        self._class_names = class_names
        self._channels = channels
        self._pool_every = pool_every
        self._hidden_dims = hidden_dims
        self._num_epochs = num_epochs
        self._optimizer_class = optimizer_class
        self._loss_function = loss_function
        self._conv_kernel_size = conv_kernel_size
        self._pooling_type = pooling_type
        self._pool_kernel_size = pool_kernel_size
        self._image_normalization_mean = image_normalization_mean
        self._image_normalization_std = image_normalization_std
        self._num_dataloader_workers = num_dataloader_workers
        self._batch_size = batch_size
        self._activation = activation
        self._use_batchnorm = use_batchnorm
        self._dropout_probability = dropout_probability
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._early_stopping_patience = early_stopping_patience
        self._number_of_augmented_copies_per_image = number_of_augmented_copies_per_image
        self._augmentation_rotation_max_degrees = augmentation_rotation_max_degrees
        self._augmentation_brightness_jitter = augmentation_brightness_jitter
        self._augmentation_contrast_jitter = augmentation_contrast_jitter
        self._augmentation_saturation_jitter = augmentation_saturation_jitter

        _, image_height, image_width = in_size
        self._image_transforms = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std),
        ])

        self._online_augmentation_sequence = [
            HorizontalFlipAugmentation(),
            RotationAugmentation(max_rotation_degrees=augmentation_rotation_max_degrees),
            ColorJitterAugmentation(
                brightness_jitter=augmentation_brightness_jitter,
                contrast_jitter=augmentation_contrast_jitter,
                saturation_jitter=augmentation_saturation_jitter,
            ),
            GaussianBlurAugmentation(),
        ]

    def run(
        self,
        train_dataset_path: pathlib.Path,
        val_dataset_path: pathlib.Path,
        test_dataset_path: pathlib.Path,
    ) -> dict:
        """
        Runs the full K-fold cross-validation pipeline.

        Combines the train and validation splits into a single pool, divides
        it into K stratified folds, trains and evaluates K models, then trains
        a final model on the full pool and tests it on the held-out test set.

        :param train_dataset_path: Path to the training split folder.
        :param val_dataset_path: Path to the validation split folder.
        :param test_dataset_path: Path to the test split folder.
        :return: Dictionary containing per-fold results, aggregate statistics,
            and final test metrics.
        """
        train_image_folder = ImageFolder(root=str(train_dataset_path), transform=None)
        val_image_folder = ImageFolder(root=str(val_dataset_path), transform=None)

        if train_image_folder.class_to_idx != val_image_folder.class_to_idx:
            raise ValueError(
                "The train and val directories have inconsistent class mappings. "
                f"Train: {train_image_folder.class_to_idx}  "
                f"Val: {val_image_folder.class_to_idx}"
            )

        all_samples: List[Tuple[str, int]] = train_image_folder.samples + val_image_folder.samples
        all_labels: List[int] = [label for _, label in all_samples]

        fold_index_groups = self._create_stratified_fold_index_groups(all_labels, self._num_folds)

        per_fold_results = []

        for fold_number in range(self._num_folds):
            print(f"\nFold [{fold_number + 1} / {self._num_folds}]")
            print("-" * 70)

            validation_sample_list = [all_samples[sample_index] for sample_index in fold_index_groups[fold_number]]
            training_sample_list = [
                all_samples[sample_index]
                for fold_index, fold_group in enumerate(fold_index_groups)
                for sample_index in fold_group
                if fold_index != fold_number
            ]

            fold_train_loader = self._build_training_data_loader(training_sample_list)
            fold_val_loader = self._build_validation_data_loader(validation_sample_list)

            fold_model = self._build_fresh_cnn()
            fold_model.train_on_data_loaders(fold_train_loader, fold_val_loader)
            fold_val_results = fold_model.validate_on_data_loader(fold_val_loader)

            per_fold_results.append({
                "fold": fold_number + 1,
                "val_loss": fold_val_results["val_loss"],
                "val_accuracy": fold_val_results["val_accuracy"],
            })

            print(
                f"\n  Fold {fold_number + 1} result: "
                f"val_loss: {fold_val_results['val_loss']:.4f}  "
                f"val_accuracy: {fold_val_results['val_accuracy'] * 100:.2f}%"
            )

        aggregate_metrics = self._compute_aggregate_metrics(per_fold_results)

        print("\n" + "=" * 70)
        print("Cross-Validation Summary")
        print("=" * 70)
        print(f"  {'Fold':>6}    {'Val Loss':>10}    {'Val Accuracy':>14}")
        print("  " + "-" * 40)
        for fold_result in per_fold_results:
            print(
                f"  {fold_result['fold']:>6}"
                f"    {fold_result['val_loss']:>10.4f}"
                f"    {fold_result['val_accuracy'] * 100:>13.2f}%"
            )
        print("  " + "-" * 40)
        print(
            f"  {'Mean':>6}"
            f"    {aggregate_metrics['mean_val_loss']:>10.4f} ± {aggregate_metrics['std_val_loss']:.4f}"
            f"    {aggregate_metrics['mean_val_accuracy'] * 100:>9.2f}% ± {aggregate_metrics['std_val_accuracy'] * 100:.2f}%"
        )
        print("=" * 70)

        print("\nTraining final model on the full train+val pool...")
        print("-" * 70)
        final_train_loader = self._build_training_data_loader(all_samples)
        final_model = self._build_fresh_cnn()
        final_model.train_on_data_loaders(final_train_loader, val_data_loader=None)

        print("\n" + "=" * 70)
        print("Testing")
        print("=" * 70)
        test_results = final_model.test_on_dataset(test_dataset_path)

        print(f"  Test accuracy: {test_results['test_accuracy'] * 100:.2f}%")
        print()
        print(f"  {'Class':<25} {'Samples':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("  " + "-" * 63)
        for class_name, class_metrics in test_results["per_class_results"].items():
            print(
                f"  {class_name:<25}"
                f"  {class_metrics['total_samples_tested']:>6}"
                f"  {class_metrics['precision']:>9.4f}"
                f"  {class_metrics['recall']:>7.4f}"
                f"  {class_metrics['f1']:>7.4f}"
            )

        return {
            "cross_validation": {
                "num_folds": self._num_folds,
                "per_fold_results": per_fold_results,
                "mean_val_loss": aggregate_metrics["mean_val_loss"],
                "std_val_loss": aggregate_metrics["std_val_loss"],
                "mean_val_accuracy": aggregate_metrics["mean_val_accuracy"],
                "std_val_accuracy": aggregate_metrics["std_val_accuracy"],
            },
            "final_model": final_model,
            "test_results": test_results,
        }

    def _create_stratified_fold_index_groups(
        self,
        all_labels: List[int],
        num_folds: int,
    ) -> List[List[int]]:
        """
        Divides sample indices into K stratified fold groups.

        Within each class, indices are shuffled and distributed round-robin
        across the K folds so that each fold's validation set contains a
        proportional representation of every class.

        :param all_labels: List of integer class labels, one per sample.
        :param num_folds: Number of folds to create.
        :return: List of K lists, where each inner list holds the sample indices
            assigned to that fold as the held-out validation set.
        :raises ValueError: If num_folds exceeds the number of samples in any class.
        """
        class_label_to_sample_indices: dict = collections.defaultdict(list)
        for sample_index, label in enumerate(all_labels):
            class_label_to_sample_indices[label].append(sample_index)

        minimum_class_count = min(len(indices) for indices in class_label_to_sample_indices.values())
        if num_folds > minimum_class_count:
            raise ValueError(
                f"num_folds ({num_folds}) exceeds the number of samples in the smallest class "
                f"({minimum_class_count}). Reduce num_folds or gather more data."
            )

        fold_index_groups: List[List[int]] = [[] for _ in range(num_folds)]
        for class_label, sample_indices_for_class in class_label_to_sample_indices.items():
            shuffled_indices = list(sample_indices_for_class)
            random.shuffle(shuffled_indices)
            for position, sample_index in enumerate(shuffled_indices):
                fold_index_groups[position % num_folds].append(sample_index)

        return fold_index_groups

    def _build_training_data_loader(
        self,
        sample_list: List[Tuple[str, int]],
    ) -> DataLoader:
        """
        Builds a DataLoader for a training split, including online augmentation.

        :param sample_list: List of (file_path, label_index) tuples for this split.
        :return: A shuffled DataLoader wrapping an AugmentedSampleListDataset.
        """
        augmented_training_dataset = AugmentedSampleListDataset(
            sample_list=sample_list,
            base_transform=self._image_transforms,
            augmentation_sequence=self._online_augmentation_sequence,
            number_of_augmented_copies=self._number_of_augmented_copies_per_image,
        )
        return DataLoader(
            augmented_training_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_dataloader_workers,
            pin_memory=torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda",
        )

    def _build_validation_data_loader(
        self,
        sample_list: List[Tuple[str, int]],
    ) -> DataLoader:
        """
        Builds a DataLoader for a validation split, without augmentation.

        :param sample_list: List of (file_path, label_index) tuples for this split.
        :return: An unshuffled DataLoader wrapping a SampleListDataset.
        """
        validation_dataset = SampleListDataset(
            sample_list=sample_list,
            transform=self._image_transforms,
        )
        return DataLoader(
            validation_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_dataloader_workers,
            pin_memory=torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda",
        )

    def _build_fresh_cnn(self) -> CNN:
        """
        Constructs a new CNN instance with freshly initialised weights.

        Each call creates an entirely new model and optimizer, ensuring that
        folds are trained independently with no shared state.

        :return: A newly initialised CNN ready for training.
        """
        return CNN(
            in_size=self._in_size,
            class_names=self._class_names,
            channels=self._channels,
            pool_every=self._pool_every,
            hidden_dims=self._hidden_dims,
            num_epochs=self._num_epochs,
            optimizer_class=self._optimizer_class,
            loss_function=self._loss_function,
            conv_kernel_size=self._conv_kernel_size,
            pooling_type=self._pooling_type,
            pool_kernel_size=self._pool_kernel_size,
            image_normalization_mean=self._image_normalization_mean,
            image_normalization_std=self._image_normalization_std,
            num_dataloader_workers=self._num_dataloader_workers,
            batch_size=self._batch_size,
            activation=self._activation,
            use_batchnorm=self._use_batchnorm,
            dropout_probability=self._dropout_probability,
            learning_rate=self._learning_rate,
            weight_decay=self._weight_decay,
            early_stopping_patience=self._early_stopping_patience,
            number_of_augmented_copies_per_image=self._number_of_augmented_copies_per_image,
            augmentation_rotation_max_degrees=self._augmentation_rotation_max_degrees,
            augmentation_brightness_jitter=self._augmentation_brightness_jitter,
            augmentation_contrast_jitter=self._augmentation_contrast_jitter,
            augmentation_saturation_jitter=self._augmentation_saturation_jitter,
        )

    def _compute_aggregate_metrics(
        self,
        per_fold_results: List[dict],
    ) -> dict:
        """
        Computes mean and standard deviation of validation metrics across all folds.

        :param per_fold_results: List of per-fold result dicts, each containing
            'val_loss' and 'val_accuracy' keys.
        :return: Dictionary with mean and std for both val_loss and val_accuracy.
        """
        all_val_losses = [fold_result["val_loss"] for fold_result in per_fold_results]
        all_val_accuracies = [fold_result["val_accuracy"] for fold_result in per_fold_results]

        return {
            "mean_val_loss": statistics.mean(all_val_losses),
            "std_val_loss": statistics.stdev(all_val_losses),
            "mean_val_accuracy": statistics.mean(all_val_accuracies),
            "std_val_accuracy": statistics.stdev(all_val_accuracies),
        }
