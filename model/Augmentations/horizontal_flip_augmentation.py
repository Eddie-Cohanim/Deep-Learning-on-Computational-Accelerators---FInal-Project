from PIL import Image
import torchvision.transforms.functional as transforms_functional

from .augmentation import Augmentation


class HorizontalFlipAugmentation(Augmentation):
    """Augmentation that mirrors images horizontally."""

    def __init__(self, augmentation_fraction: float = 0.5) -> None:
        """
        :param augmentation_fraction: Fraction of eligible images to augment.
        """
        super().__init__(augmentation_fraction=augmentation_fraction)

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        return transforms_functional.hflip(image)

    def _augmentation_name(self) -> str:
        return "hflip"
