import collections
import json
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
    """
    Orchestrates stratified K-fold cross-validation for any image classifier.

    The model to train is supplied as a factory callable that takes no arguments
    and returns a freshly initialised model. This decouples the cross-validator
    from any specific architecture — it works with CNN, PretrainedModel, or any
    future classifier that implements the same training interface.

    The combined train and validation splits are divided into K folds. For each
    fold, a fresh model is trained on the K-1 remaining folds and evaluated on
    the held-out fold. After all folds are complete, a final model is trained on
    the full train+val pool and evaluated on the held-out test set.

    Results include per-fold metrics and aggregate mean ± standard deviation for
    both validation loss and validation accuracy.
    """

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
        experiment_folder_path: pathlib.Path = None,
        save_best_checkpoint: bool = False,
        resume_fold_state: dict = None,
    ) -> dict:
        """
        Runs the full K-fold cross-validation pipeline.

        Combines the train and validation splits into a single pool, divides it into
        K stratified folds, trains and evaluates K models, then trains a final model
        on the full pool and evaluates it on the held-out test set.

        :param train_dataset_path: Path to the training split folder.
        :param val_dataset_path: Path to the validation split folder.
        :param test_dataset_path: Path to the test split folder.
        :param experiment_folder_path: Optional path to the versioned results folder.
            When provided, fold state is persisted to disk after each fold so the run
            can be resumed if the process is interrupted.
        :param save_best_checkpoint: When True, saves the best weights for each fold
            to disk whenever a new best validation loss is reached during training.
            Requires experiment_folder_path to be set.
        :param resume_fold_state: Optional dict loaded from a previously saved
            fold_state.json. When provided, fold splits and completed results are
            restored from this state and training resumes from the next unfinished fold.
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

        if resume_fold_state is not None:
            fold_index_groups = resume_fold_state["fold_index_groups"]
            per_fold_results = list(resume_fold_state["completed_folds"])
            starting_fold_index = resume_fold_state["next_fold_index"]
        else:
            fold_index_groups = self._create_stratified_fold_index_groups(all_labels, self._num_folds)
            per_fold_results = []
            starting_fold_index = 0
            if experiment_folder_path is not None:
                self._save_fold_state(experiment_folder_path, fold_index_groups, [], 0)

        for fold_number in range(starting_fold_index, self._num_folds):
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

            fold_best_checkpoint_path = None
            if save_best_checkpoint and experiment_folder_path is not None:
                fold_best_checkpoint_path = experiment_folder_path / f"fold_{fold_number + 1}_best_checkpoint.pth"

            fold_model = self._model_factory()
            fold_training_result = fold_model.train_on_data_loaders(
                fold_train_loader,
                fold_val_loader,
                best_checkpoint_path=fold_best_checkpoint_path,
            )
            fold_val_results = fold_model.validate_on_data_loader(fold_val_loader)

            del fold_model
            torch.cuda.empty_cache()

            per_fold_results.append({
                "fold": fold_number + 1,
                "val_loss": fold_val_results["val_loss"],
                "val_accuracy": fold_val_results["val_accuracy"],
                "epochs_trained": fold_training_result["epochs_trained"],
            })

            if experiment_folder_path is not None:
                self._save_fold_state(
                    experiment_folder_path,
                    fold_index_groups,
                    per_fold_results,
                    fold_number + 1,
                )

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

        max_fold_epochs = max(fold_result["epochs_trained"] for fold_result in per_fold_results)
        final_num_epochs = int(max_fold_epochs * 1.10)
        print(f"\n  Max fold epochs: {max_fold_epochs}  →  Final model will train for {final_num_epochs} epochs (+10%)")

        torch.cuda.empty_cache()
        print("\nTraining final model on the full train+val pool...")
        print("-" * 70)
        final_train_loader = self._build_training_data_loader(all_samples)
        final_model = self._model_factory()
        final_model.train_on_data_loaders(
            final_train_loader,
            val_data_loader=None,
            num_epochs_override=final_num_epochs,
        )

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

    def _save_fold_state(
        self,
        experiment_folder_path: pathlib.Path,
        fold_index_groups: List[List[int]],
        completed_folds: List[dict],
        next_fold_index: int,
    ) -> None:
        """
        Writes fold state to disk so an interrupted run can resume from the correct fold.

        The file records the exact fold index groups so the same train/val splits are
        reproduced on restart without relying on random seed replay.

        :param experiment_folder_path: Versioned results folder where the file is written.
        :param fold_index_groups: List of K lists, each holding sample indices for one fold.
        :param completed_folds: Per-fold result dicts accumulated so far.
        :param next_fold_index: Zero-based index of the fold that should run next.
        """
        fold_state = {
            "num_folds": self._num_folds,
            "fold_index_groups": fold_index_groups,
            "completed_folds": completed_folds,
            "next_fold_index": next_fold_index,
        }
        fold_state_path = experiment_folder_path / "fold_state.json"
        with fold_state_path.open("w", encoding="utf-8") as fold_state_file:
            json.dump(fold_state, fold_state_file, indent=4)

    def _create_stratified_fold_index_groups(
        self,
        all_labels: List[int],
        num_folds: int,
    ) -> List[List[int]]:
        """
        Divides sample indices into K stratified fold groups.

        Within each class, indices are shuffled and distributed round-robin across
        the K folds so that each fold's validation set has proportional class coverage.

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

        GPU augmentation is applied inside the model's training loop, so no
        augmentation wrapper is needed at the dataset level.

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
            persistent_workers=self._num_dataloader_workers > 0,
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
            persistent_workers=self._num_dataloader_workers > 0,
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
