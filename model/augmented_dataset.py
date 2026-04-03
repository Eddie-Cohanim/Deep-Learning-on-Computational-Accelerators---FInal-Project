import pathlib
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model.Augmentations.augmentation import Augmentation


class AugmentedTrainingDataset(Dataset):
    """
    Wraps an ImageFolder dataset and exposes augmented copies of each image
    on the fly during training, without writing anything to disk.

    For each original image, the dataset exposes:
        - 1 copy with the base transform (resize + normalize only)
        - number_of_augmented_copies copies with the full augmentation sequence
          applied before the base transform

    The effective length is:
        len(base_dataset) * (1 + number_of_augmented_copies)

    Because the augmentations that involve random parameters (rotation,
    color jitter, gaussian blur) re-sample their randomness on every call,
    each augmented copy of the same image will differ from the others.
    """

    def __init__(
        self,
        base_dataset: ImageFolder,
        base_transform: transforms.Compose,
        augmentation_sequence: List[Augmentation],
        number_of_augmented_copies: int,
    ) -> None:
        """
        :param base_dataset: An ImageFolder loaded with transform=None so that
            raw PIL images are returned. The dataset's .samples list is used
            directly to load images.
        :param base_transform: Transform applied to every image after optional
            augmentation. Should contain only resize, ToTensor, and Normalize —
            no random operations.
        :param augmentation_sequence: Ordered list of Augmentation instances
            whose apply_to_image() method is called in sequence for each
            augmented copy. The original copy (copy_number == 0) skips this
            sequence entirely.
        :param number_of_augmented_copies: Number of augmented versions to
            generate per original image. The dataset length is multiplied by
            (1 + number_of_augmented_copies).
        """
        self._base_dataset = base_dataset
        self._base_transform = base_transform
        self._augmentation_sequence = augmentation_sequence
        self._number_of_augmented_copies = number_of_augmented_copies
        self._total_copies_per_original = 1 + number_of_augmented_copies

    def __len__(self) -> int:
        return len(self._base_dataset) * self._total_copies_per_original

    def __getitem__(self, index: int):
        original_index = index // self._total_copies_per_original
        copy_number = index % self._total_copies_per_original

        image_file_path, label = self._base_dataset.samples[original_index]

        with Image.open(image_file_path) as opened_image:
            pil_image = opened_image.convert("RGB")

        if copy_number == 0:
            return self._base_transform(pil_image), label

        for augmentation in self._augmentation_sequence:
            pil_image = augmentation.apply_to_image(pil_image)

        return self._base_transform(pil_image), label
