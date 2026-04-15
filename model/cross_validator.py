import collections
import pathlib
import random
import statistics
from typing import Callable, List, Sequence, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model.augmented_dataset import SampleListDataset


class CrossValidator:
    """Orchestrates stratified K-fold cross-validation for any image classifier."""

    def __init__(
        self,
        num_folds: int,
        in_size: tuple,
        image_normalization_mean: list,
        image_normalization_std: list,
        batch_size: int,
        num_dataloader_workers: int,
        model_factory: Callable,
    ) -> None:
        """
        :param num_folds: Number of folds K. Must be at least 2 and no greater than
            the number of samples in the smallest class.
        :param in_size: Input image shape as (channels, height, width). Used to build
            the image resize transform for cross-validation DataLoaders.
        :param image_normalization_mean: Per-channel mean for input normalisation.
        :param image_normalization_std: Per-channel std for input normalisation.
        :param batch_size: Number of images per batch for cross-validation DataLoaders.
        :param num_dataloader_workers: Number of DataLoader worker processes.
        :param model_factory: Zero-argument callable that returns a freshly initialised
            model ready for training. Called once per fold and once for the final model.
            The returned model must implement train_on_data_loaders, validate_on_data_loader,
            and test_on_dataset.
        """
        self._num_folds = num_folds
        self._batch_size = batch_size
        self._num_dataloader_workers = num_dataloader_workers
        self._model_factory = model_factory

        _, image_height, image_width = in_size
        self._image_transforms = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std),
        ])

    def run(
        self,
        train_dataset_path: pathlib.Path,
        val_dataset_path: pathlib.Path,
        test_dataset_path: pathlib.Path,
    ) -> dict:
        """
        Runs the full K-fold cross-validation pipeline.

        :param train_dataset_path: Path to the training split folder.
        :param val_dataset_path: Path to the validation split folder.
        :param test_dataset_path: Path to the test split folder.
        :return: Dictionary containing per-fold results, aggregate statistics, and final test metrics.
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

            fold_model = self._model_factory()
            fold_model.train_on_data_loaders(fold_train_loader, fold_val_loader)
            fold_val_results = fold_model.validate_on_data_loader(fold_val_loader)

            del fold_model
            torch.cuda.empty_cache()

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

        torch.cuda.empty_cache()
        print("\nTraining final model on the full train+val pool...")
        print("-" * 70)
        final_train_loader = self._build_training_data_loader(all_samples)
        final_model = self._model_factory()
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

        :param all_labels: List of integer class labels, one per sample.
        :param num_folds: Number of folds to create.
        :return: List of K lists, each holding the sample indices for one validation fold.
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
        Builds a shuffled DataLoader for a training split.

        :param sample_list: List of (file_path, label_index) tuples for this split.
        :return: A shuffled DataLoader over the training samples.
        """
        training_dataset = SampleListDataset(
            sample_list=sample_list,
            transform=self._image_transforms,
        )
        return DataLoader(
            training_dataset,
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
        Builds an unshuffled DataLoader for a validation split.

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

    def _compute_aggregate_metrics(
        self,
        per_fold_results: List[dict],
    ) -> dict:
        """
        Computes mean and standard deviation of validation metrics across all folds.

        :param per_fold_results: List of per-fold result dicts with 'val_loss' and
            'val_accuracy' keys.
        :return: Dict with mean and std for both val_loss and val_accuracy.
        """
        all_val_losses = [fold_result["val_loss"] for fold_result in per_fold_results]
        all_val_accuracies = [fold_result["val_accuracy"] for fold_result in per_fold_results]

        return {
            "mean_val_loss": statistics.mean(all_val_losses),
            "std_val_loss": statistics.stdev(all_val_losses),
            "mean_val_accuracy": statistics.mean(all_val_accuracies),
            "std_val_accuracy": statistics.stdev(all_val_accuracies),
        }
