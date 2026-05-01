import collections
import json
import pathlib
import random
import statistics
from typing import Callable, List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class SampleListDataset(Dataset):
    def __init__(self, sample_list: List[Tuple[str, int]], transform: transforms.Compose) -> None:
        self._sample_list = sample_list
        self._transform = transform

    def __len__(self) -> int:
        return len(self._sample_list)

    def __getitem__(self, index: int):
        image_file_path, label = self._sample_list[index]
        with Image.open(image_file_path) as opened_image:
            pil_image = opened_image.convert("RGB")
        return self._transform(pil_image), label


class CrossValidator:
    def __init__(
        self,
        num_folds: int,
        in_size: tuple,
        image_normalization_mean: list,
        image_normalization_std: list,
        batch_size: int,
        num_workers: int,
        model_factory: Callable,
    ) -> None:
        self._num_folds = num_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
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
        train_folder = ImageFolder(root=str(train_dataset_path), transform=None)
        val_folder = ImageFolder(root=str(val_dataset_path), transform=None)

        if train_folder.class_to_idx != val_folder.class_to_idx:
            raise ValueError(
                "The train and val directories have inconsistent class mappings. "
                f"Train: {train_folder.class_to_idx}  "
                f"Val: {val_folder.class_to_idx}"
            )

        samples = train_folder.samples + val_folder.samples
        labels = [label for _, label in samples]

        if resume_fold_state is not None:
            fold_groups = resume_fold_state["fold_index_groups"]
            fold_results = list(resume_fold_state["completed_folds"])
            start_fold = resume_fold_state["next_fold_index"]
        else:
            fold_groups = self._create_stratified_fold_index_groups(labels, self._num_folds)
            fold_results = []
            start_fold = 0
            if experiment_folder_path is not None:
                self._save_fold_state(experiment_folder_path, fold_groups, [], 0)

        for fold_number in range(start_fold, self._num_folds):
            print(f"\nFold [{fold_number + 1} / {self._num_folds}]")
            print("-" * 70)

            val_samples = [samples[idx] for idx in fold_groups[fold_number]]
            train_samples = [
                samples[idx]
                for fold_idx, fold_group in enumerate(fold_groups)
                for idx in fold_group
                if fold_idx != fold_number
            ]

            train_loader = self._build_training_data_loader(train_samples)
            val_loader = self._build_validation_data_loader(val_samples)

            checkpoint_path = None
            if save_best_checkpoint and experiment_folder_path is not None:
                checkpoint_path = experiment_folder_path / f"fold_{fold_number + 1}_best_checkpoint.pth"

            fold_model = self._model_factory()
            train_result = fold_model.train_on_data_loaders(
                train_loader,
                val_loader,
                best_checkpoint_path=checkpoint_path,
            )
            val_result = fold_model.validate_on_data_loader(val_loader)

            del fold_model
            torch.cuda.empty_cache()

            fold_results.append({
                "fold": fold_number + 1,
                "val_loss": val_result["val_loss"],
                "val_accuracy": val_result["val_accuracy"],
                "epochs_trained": train_result["epochs_trained"],
            })

            if experiment_folder_path is not None:
                self._save_fold_state(
                    experiment_folder_path,
                    fold_groups,
                    fold_results,
                    fold_number + 1,
                )

            print(
                f"\n  Fold {fold_number + 1} result: "
                f"val_loss: {val_result['val_loss']:.4f}  "
                f"val_accuracy: {val_result['val_accuracy'] * 100:.2f}%"
            )

        metrics = self._compute_aggregate_metrics(fold_results)

        print("\n" + "=" * 70)
        print("Cross-Validation Summary")
        print("=" * 70)
        print(f"  {'Fold':>6}    {'Val Loss':>10}    {'Val Accuracy':>14}")
        print("  " + "-" * 40)
        for result in fold_results:
            print(
                f"  {result['fold']:>6}"
                f"    {result['val_loss']:>10.4f}"
                f"    {result['val_accuracy'] * 100:>13.2f}%"
            )
        print("  " + "-" * 40)
        print(
            f"  {'Mean':>6}"
            f"    {metrics['mean_val_loss']:>10.4f} ± {metrics['std_val_loss']:.4f}"
            f"    {metrics['mean_val_accuracy'] * 100:>9.2f}% ± {metrics['std_val_accuracy'] * 100:.2f}%"
        )
        print("=" * 70)

        max_epochs = max(result["epochs_trained"] for result in fold_results)
        final_epochs = int(max_epochs * 1.10)
        print(f"\n  Max fold epochs: {max_epochs}  →  Final model will train for {final_epochs} epochs (+10%)")

        torch.cuda.empty_cache()
        print("\nTraining final model on the full train+val pool...")
        print("-" * 70)
        train_loader = self._build_training_data_loader(samples)
        final_model = self._model_factory()
        final_model.train_on_data_loaders(
            train_loader,
            val_data_loader=None,
            num_epochs_override=final_epochs,
        )

        print("\n" + "=" * 70)
        print("Testing")
        print("=" * 70)
        test_result = final_model.test_on_dataset(test_dataset_path)

        print(f"  Test accuracy: {test_result['test_accuracy'] * 100:.2f}%")
        print()
        print(f"  {'Class':<25} {'Samples':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("  " + "-" * 63)
        for class_name, class_metrics in test_result["per_class_results"].items():
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
                "per_fold_results": fold_results,
                "mean_val_loss": metrics["mean_val_loss"],
                "std_val_loss": metrics["std_val_loss"],
                "mean_val_accuracy": metrics["mean_val_accuracy"],
                "std_val_accuracy": metrics["std_val_accuracy"],
            },
            "final_model": final_model,
            "test_results": test_result,
        }

    def _save_fold_state(
        self,
        experiment_folder_path: pathlib.Path,
        fold_groups: List[List[int]],
        completed_folds: List[dict],
        next_fold_index: int,
    ) -> None:
        state = {
            "num_folds": self._num_folds,
            "fold_index_groups": fold_groups,
            "completed_folds": completed_folds,
            "next_fold_index": next_fold_index,
        }
        state_path = experiment_folder_path / "fold_state.json"
        with state_path.open("w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=4)

    def _create_stratified_fold_index_groups(
        self,
        labels: List[int],
        num_folds: int,
    ) -> List[List[int]]:
        class_to_indices = collections.defaultdict(list)
        for index, label in enumerate(labels):
            class_to_indices[label].append(index)

        min_class_count = min(len(indices) for indices in class_to_indices.values())
        if num_folds > min_class_count:
            raise ValueError(
                f"num_folds ({num_folds}) exceeds the number of samples in the smallest class "
                f"({min_class_count}). Reduce num_folds or gather more data."
            )

        fold_groups = [[] for _ in range(num_folds)]
        for _, class_indices in class_to_indices.items():
            shuffled = list(class_indices)
            random.shuffle(shuffled)
            for position, index in enumerate(shuffled):
                fold_groups[position % num_folds].append(index)

        return fold_groups

    def _build_training_data_loader(
        self,
        sample_list: List[Tuple[str, int]],
    ) -> DataLoader:
        # augmentation is applied in the model's training loop, not here
        dataset = SampleListDataset(
            sample_list=sample_list,
            transform=self._image_transforms,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self._num_workers > 0,
        )

    def _build_validation_data_loader(
        self,
        sample_list: List[Tuple[str, int]],
    ) -> DataLoader:
        dataset = SampleListDataset(
            sample_list=sample_list,
            transform=self._image_transforms,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self._num_workers > 0,
        )

    def _compute_aggregate_metrics(
        self,
        fold_results: List[dict],
    ) -> dict:
        losses = [result["val_loss"] for result in fold_results]
        accuracies = [result["val_accuracy"] for result in fold_results]

        return {
            "mean_val_loss": statistics.mean(losses),
            "std_val_loss": statistics.stdev(losses),
            "mean_val_accuracy": statistics.mean(accuracies),
            "std_val_accuracy": statistics.stdev(accuracies),
        }