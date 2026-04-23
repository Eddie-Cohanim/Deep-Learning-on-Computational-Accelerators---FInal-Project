import argparse
import pathlib
import random
import shutil

_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class DatasetSplitter:
    """
    Splits a folder of class-labelled images into train, validation, and test sets.

    Expects the source folder to contain one subfolder per class, each holding
    image files:
        <source_folder>/<class_name>/<image_file>

    Images are shuffled randomly before splitting. The train percentage is
    specified by the caller; the remainder is divided evenly between val and test.
    Files are copied to the destination folders, leaving the source untouched.
    """

    def __init__(
        self,
        source_folder_path: pathlib.Path,
        train_destination_path: pathlib.Path,
        val_destination_path: pathlib.Path,
        test_destination_path: pathlib.Path,
        train_percentage: int,
    ) -> None:
        """
        :param source_folder_path: Folder containing one subfolder per class.
        :param train_destination_path: Root destination folder for training images.
        :param val_destination_path: Root destination folder for validation images.
        :param test_destination_path: Root destination folder for test images.
        :param train_percentage: Integer percentage of images assigned to train (e.g. 80).
            The remaining percentage is split evenly between val and test.
        :raises ValueError: If train_percentage is not between 1 and 98 inclusive.
        """
        if not 1 <= train_percentage <= 98:
            raise ValueError(
                f"train_percentage must be between 1 and 98, got {train_percentage}."
            )

        self._source_folder_path = source_folder_path
        self._train_destination_path = train_destination_path
        self._val_destination_path = val_destination_path
        self._test_destination_path = test_destination_path
        self._train_fraction = train_percentage / 100.0
        self._remaining_fraction = (1.0 - self._train_fraction) / 2.0

    def split(self) -> None:
        """
        Performs the split across all class subfolders found in the source folder.

        For each class subfolder, all image files are collected, shuffled, then
        divided into train / val / test according to the configured fractions.
        Each image is copied into the corresponding destination subfolder.

        Destination class subfolders are created automatically if they do not exist.
        """
        class_subfolders = [
            entry for entry in self._source_folder_path.iterdir()
            if entry.is_dir()
        ]

        if not class_subfolders:
            print(f"No class subfolders found in {self._source_folder_path}. Nothing to split.")
            return

        for class_folder_path in sorted(class_subfolders):
            class_name = class_folder_path.name
            all_image_paths = [
                file_path
                for file_path in class_folder_path.rglob("*")
                if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
            ]

            if not all_image_paths:
                print(f"  [{class_name}] No images found. Skipping.")
                continue

            random.shuffle(all_image_paths)

            total_image_count = len(all_image_paths)
            train_cutoff_index = int(total_image_count * self._train_fraction)
            val_cutoff_index = train_cutoff_index + int(total_image_count * self._remaining_fraction)

            train_image_paths = all_image_paths[:train_cutoff_index]
            val_image_paths = all_image_paths[train_cutoff_index:val_cutoff_index]
            test_image_paths = all_image_paths[val_cutoff_index:]

            self._copy_images_to_destination(train_image_paths, self._train_destination_path / class_name)
            self._copy_images_to_destination(val_image_paths, self._val_destination_path / class_name)
            self._copy_images_to_destination(test_image_paths, self._test_destination_path / class_name)

            print(
                f"  [{class_name}]  "
                f"train: {len(train_image_paths)}  "
                f"val: {len(val_image_paths)}  "
                f"test: {len(test_image_paths)}"
            )

    def _copy_images_to_destination(
        self,
        image_paths: list,
        destination_folder_path: pathlib.Path,
    ) -> None:
        """
        Copies a list of image files into the given destination folder.

        Creates the destination folder if it does not already exist.

        :param image_paths: List of source image file paths to copy.
        :param destination_folder_path: Folder to copy the images into.
        """
        destination_folder_path.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            shutil.copy2(image_path, destination_folder_path / image_path.name)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description=(
            "Split a folder of class-labelled images into train, val, and test sets. "
            "The source folder must contain one subfolder per class. "
            "The train percentage is specified; the remainder is split evenly between val and test."
        )
    )
    argument_parser.add_argument(
        "--source",
        required=True,
        help="Path to the source folder containing one subfolder per class.",
    )
    argument_parser.add_argument(
        "--train",
        required=True,
        help="Destination path for training images.",
    )
    argument_parser.add_argument(
        "--val",
        required=True,
        help="Destination path for validation images.",
    )
    argument_parser.add_argument(
        "--test",
        required=True,
        help="Destination path for test images.",
    )
    argument_parser.add_argument(
        "--train-percent",
        type=int,
        required=True,
        help="Integer percentage of images to assign to the training set (e.g. 80).",
    )

    parsed_arguments = argument_parser.parse_args()

    splitter = DatasetSplitter(
        source_folder_path=pathlib.Path(parsed_arguments.source),
        train_destination_path=pathlib.Path(parsed_arguments.train),
        val_destination_path=pathlib.Path(parsed_arguments.val),
        test_destination_path=pathlib.Path(parsed_arguments.test),
        train_percentage=parsed_arguments.train_percent,
    )

    print(f"Splitting dataset from: {parsed_arguments.source}")
    print(f"  Train ({parsed_arguments.train_percent}%) -> {parsed_arguments.train}")
    print(f"  Val   ({(100 - parsed_arguments.train_percent) // 2}%) -> {parsed_arguments.val}")
    print(f"  Test  ({(100 - parsed_arguments.train_percent) // 2}%) -> {parsed_arguments.test}")
    print()

    splitter.split()

    print("\nDone.")
