import json
import pathlib

from model.Augmentations.augmentations import _SUPPORTED_IMAGE_EXTENSIONS


class DatasetValidationReport:
    """Holds the results of a dataset structure validation pass."""

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
    """Orchestrates dataset validation for a structured image classification dataset."""

    def __init__(
        self,
        dataset_root_path: pathlib.Path,
        config_file_path: pathlib.Path,
    ) -> None:
        """
        :param dataset_root_path: Root folder containing the train/, val/, and
            test/ splits as immediate subdirectories.
        :param config_file_path: Path to config.json. Must contain a "model" key
            with "class_names".
        :raises FileNotFoundError: If config_file_path does not exist.
        :raises KeyError: If expected keys are missing from config.json.
        """
        with config_file_path.open(encoding="utf-8") as config_file:
            loaded_config = json.load(config_file)

        model_config = loaded_config["model"]

        self._training_split_path = dataset_root_path / "train"
        self._validation_split_path = dataset_root_path / "val"
        self._test_split_path = dataset_root_path / "test"

        self._class_names: list = model_config["class_names"]

    def validate_dataset(self) -> DatasetValidationReport:
        """
        Verifies that the dataset directory structure matches the configured class names.

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

    def run_full_pipeline(self) -> DatasetValidationReport:
        """
        Validates the dataset structure and prints the report.

        :return: The DatasetValidationReport from the validation step.
        """
        dataset_validation_report = self.validate_dataset()
        dataset_validation_report.print_summary()
        return dataset_validation_report
