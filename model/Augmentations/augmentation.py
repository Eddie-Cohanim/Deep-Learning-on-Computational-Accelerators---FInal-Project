# NOTE: A global list of augmented files still needs to be implemented in the main pipeline.
# This list should be initialized once (e.g. as an empty list) before any augmentation
# is performed, then passed to each augment() call. The method updates it in-place so
# that no file is ever augmented twice across multiple augmentation passes.

import abc
import pathlib
import random
from PIL import Image


_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class Augmentation(abc.ABC):
    """
    Base class for offline image augmentation.

    Subclasses implement _apply_to_image() to define the specific transformation.
    The augment() method selects a random subset of eligible images from a dataset
    folder, applies the transformation, saves each result as a new file alongside
    the original, and records the new file in the shared augmented-files list.

    The augmented-files list is shared across all augmentation instances and passed
    in by the caller so that no image is ever augmented twice.
    """

    def __init__(self, augmentation_fraction: float = 0.5) -> None:
        """
        :param augmentation_fraction: Fraction of eligible images to augment,
            in the range (0.0, 1.0]. For example, 0.5 augments up to half the
            eligible images. Defaults to 0.5.
        :raises ValueError: If augmentation_fraction is not in (0.0, 1.0].
        """
        if not 0.0 < augmentation_fraction <= 1.0:
            raise ValueError(
                f"augmentation_fraction must be in (0.0, 1.0], got {augmentation_fraction}."
            )
        self._augmentation_fraction = augmentation_fraction

    def augment(
        self,
        dataset_path: pathlib.Path,
        augmented_files: list,
    ) -> list:
        """
        Applies this augmentation to a random subset of images in the dataset folder.

        Scans all class subfolders under dataset_path for image files that have not
        yet been augmented (i.e. are not in augmented_files). Selects a random number
        of those eligible files, applies the augmentation, saves each result next to
        the original with a descriptive filename suffix, and appends the new file paths
        to augmented_files.

        :param dataset_path: Root folder of the dataset. Must contain one subfolder
            per class, each holding image files:
            dataset_path/<class_name>/<image_file>
        :param augmented_files: Shared list of file paths (as strings) that have
            already been augmented. Updated in-place with any new files created.
        :return: The updated augmented_files list.
        """
        augmented_files_set = set(augmented_files)
        eligible_image_paths = [
            image_path
            for image_path in dataset_path.rglob("*")
            if image_path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
            and str(image_path) not in augmented_files_set
        ]

        if not eligible_image_paths:
            return augmented_files

        number_of_files_to_augment = max(1, int(len(eligible_image_paths) * self._augmentation_fraction))
        selected_image_paths = random.sample(eligible_image_paths, number_of_files_to_augment)

        for original_image_path in selected_image_paths:
            with Image.open(original_image_path) as raw_image:
                original_image = raw_image.convert("RGB")
            augmented_image = self._apply_to_image(original_image)

            augmented_file_name = (
                original_image_path.stem
                + "_"
                + self._augmentation_name()
                + original_image_path.suffix
            )
            augmented_image_path = original_image_path.parent / augmented_file_name
            augmented_image.save(augmented_image_path)

            augmented_files.append(str(augmented_image_path))

        return augmented_files

    @abc.abstractmethod
    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        """
        Applies the specific augmentation transformation to a single PIL image.

        :param image: The original image.
        :return: The transformed image.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _augmentation_name(self) -> str:
        """
        Returns a short descriptive name used as the filename suffix for augmented files.
        For example: 'hflip', 'rotation', 'colorjitter'.
        """
        raise NotImplementedError
