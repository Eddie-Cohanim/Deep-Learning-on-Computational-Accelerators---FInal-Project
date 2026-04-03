import json
import pathlib
from typing import Optional

from model.Augmentations.augmentation import _SUPPORTED_IMAGE_EXTENSIONS
from model.Augmentations.horizontal_flip_augmentation import HorizontalFlipAugmentation
from model.Augmentations.rotation_augmentation import RotationAugmentation
from model.Augmentations.color_jitter_augmentation import ColorJitterAugmentation


class DatasetValidationReport:
    """
    Holds the results of a dataset structure validation pass.

    Callers can inspect is_valid programmatically to decide whether to proceed,
    and call print_summary() to display the results to the user.
    """

    def __init__(
        self,
        split_name_to_class_image_counts: dict,
        missing_split_names: list,
        missing_class_directories: dict,
    ) -> None:
        """
        :param split_name_to_class_image_counts: Maps each split name
            ("train", "val", "test") to a dict of {class_name: image_count}.
        :param missing_split_names: List of split names whose directories
            were not found on disk.
        :param missing_class_directories: Maps each split name to the list
            of class names that had no corresponding subfolder.
        """
        self._split_name_to_class_image_counts = split_name_to_class_image_counts
        self._missing_split_names = missing_split_names
        self._missing_class_directories = missing_class_directories

    @property
    def is_valid(self) -> bool:
        """True if all splits and all class directories were found on disk."""
        return (
            len(self._missing_split_names) == 0
            and all(
                len(missing_classes) == 0
                for missing_classes in self._missing_class_directories.values()
            )
        )

    def print_summary(self) -> None:
        """Prints a human-readable validation summary to stdout."""
        separator = "=" * 70
        print("\nDataset Validation")
        print(separator)

        for split_name, class_image_counts in self._split_name_to_class_image_counts.items():
            split_status = "OK" if split_name not in self._missing_split_names else "MISSING"
            print(f"\n  Split: {split_name}    STATUS: {split_status}")

            if split_name in self._missing_split_names:
                print(f"    [MISSING SPLIT] {split_name}/  — directory not found")
                continue

            for class_name, image_count in class_image_counts.items():
                print(f"    {class_name:<30}    {image_count} images")

            missing_classes_in_split = self._missing_class_directories.get(split_name, [])
            for missing_class_name in missing_classes_in_split:
                print(f"    [MISSING CLASS] {split_name}/{missing_class_name}  — subfolder not found")

        print(f"\n{separator}")
        result_label = "PASSED" if self.is_valid else "FAILED"
        result_detail = "all splits and classes found" if self.is_valid else "see above for details"
        print(f"VALIDATION RESULT: {result_label}   ({result_detail})")
        print(f"{separator}\n")

    def __str__(self) -> str:
        """Returns the same text that print_summary would print."""
        import io
        import sys
        string_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = string_buffer
        self.print_summary()
        sys.stdout = original_stdout
        return string_buffer.getvalue()


class PreprocessingPipeline:
    """
    Orchestrates dataset validation and offline augmentation for a structured
    image classification dataset.

    The dataset must follow this layout:
        <dataset_root>/train/<class_name>/<image_file>
        <dataset_root>/val/<class_name>/<image_file>
        <dataset_root>/test/<class_name>/<image_file>

    All configuration is read from config.json at construction time.
    """

    def __init__(
        self,
        dataset_root_path: pathlib.Path,
        config_file_path: pathlib.Path,
    ) -> None:
        """
        :param dataset_root_path: Root folder containing the train/, val/, and
            test/ splits as immediate subdirectories.
        :param config_file_path: Path to config.json. Must contain a "model" key
            with "class_names" and a "preprocessing" key with augmentation settings.
        :raises FileNotFoundError: If config_file_path does not exist.
        :raises KeyError: If expected keys are missing from config.json.
        """
        with config_file_path.open(encoding="utf-8") as config_file:
            loaded_config = json.load(config_file)

        model_config = loaded_config["model"]
        preprocessing_config = loaded_config["preprocessing"]

        self._training_split_path = dataset_root_path / "train"
        self._validation_split_path = dataset_root_path / "val"
        self._test_split_path = dataset_root_path / "test"

        self._class_names: list = model_config["class_names"]
        self._run_augmentation: bool = preprocessing_config["run_augmentation"]

        augmentation_fraction = preprocessing_config["augmentation_fraction"]

        self._horizontal_flip_augmentation = HorizontalFlipAugmentation(
            augmentation_fraction=augmentation_fraction,
        )
        self._rotation_augmentation = RotationAugmentation(
            max_rotation_degrees=preprocessing_config["augmentation_rotation_max_degrees"],
            augmentation_fraction=augmentation_fraction,
        )
        self._color_jitter_augmentation = ColorJitterAugmentation(
            brightness_jitter=preprocessing_config["augmentation_brightness_jitter"],
            contrast_jitter=preprocessing_config["augmentation_contrast_jitter"],
            saturation_jitter=preprocessing_config["augmentation_saturation_jitter"],
            augmentation_fraction=augmentation_fraction,
        )

    def validate_dataset(self) -> DatasetValidationReport:
        """
        Verifies that the dataset directory structure matches the configured class names.

        Checks that each of train/, val/, and test/ exists, that each split contains
        a subfolder for every name in class_names, and counts image files per class.

        Does not raise on failure — all errors are captured in the returned report
        so that callers can decide how to handle them.

        :return: A DatasetValidationReport describing what was found and what was missing.
        """
        split_entries = [
            ("train", self._training_split_path),
            ("val", self._validation_split_path),
            ("test", self._test_split_path),
        ]

        split_name_to_class_image_counts = {}
        missing_split_names = []
        missing_class_directories = {}

        for split_name, split_path in split_entries:
            class_image_counts = {}
            missing_classes_in_split = []

            if not split_path.exists() or not split_path.is_dir():
                missing_split_names.append(split_name)
                split_name_to_class_image_counts[split_name] = {}
                missing_class_directories[split_name] = []
                continue

            for class_name in self._class_names:
                class_directory_path = split_path / class_name

                if not class_directory_path.exists() or not class_directory_path.is_dir():
                    missing_classes_in_split.append(class_name)
                    class_image_counts[class_name] = 0
                    continue

                image_count = sum(
                    1
                    for file_path in class_directory_path.rglob("*")
                    if file_path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
                )
                class_image_counts[class_name] = image_count

            split_name_to_class_image_counts[split_name] = class_image_counts
            missing_class_directories[split_name] = missing_classes_in_split

        return DatasetValidationReport(
            split_name_to_class_image_counts=split_name_to_class_image_counts,
            missing_split_names=missing_split_names,
            missing_class_directories=missing_class_directories,
        )

    def run_augmentation(self) -> Optional[list]:
        """
        Runs all three offline augmentations over the training split, ensuring
        no image is augmented more than once across the three passes.

        A single shared list of already-augmented file paths is initialised here
        and passed through each augmentation call in order: horizontal flip,
        rotation, color jitter.

        Only the training split is augmented. The validation and test splits
        are never touched.

        If run_augmentation is False in config, prints an informational message
        and returns None without modifying any files.

        :return: The list of all augmented file path strings created, or None if
            augmentation was disabled in config.
        """
        if not self._run_augmentation:
            print("Augmentation is disabled in config (preprocessing.run_augmentation = false). Skipping.")
            return None

        print("Running offline augmentation on training split...")

        shared_augmented_file_paths: list = []

        shared_augmented_file_paths = self._horizontal_flip_augmentation.augment(
            self._training_split_path,
            shared_augmented_file_paths,
        )
        shared_augmented_file_paths = self._rotation_augmentation.augment(
            self._training_split_path,
            shared_augmented_file_paths,
        )
        shared_augmented_file_paths = self._color_jitter_augmentation.augment(
            self._training_split_path,
            shared_augmented_file_paths,
        )

        print(f"Augmentation complete. {len(shared_augmented_file_paths)} new images created.")
        return shared_augmented_file_paths

    def run_full_pipeline(self) -> DatasetValidationReport:
        """
        Executes the complete preprocessing pipeline in sequence:
            1. Validates the dataset structure and prints the report.
            2. Runs offline augmentation on the training split (if enabled).

        :return: The DatasetValidationReport from the validation step.
        """
        dataset_validation_report = self.validate_dataset()
        dataset_validation_report.print_summary()

        self.run_augmentation()

        return dataset_validation_report
