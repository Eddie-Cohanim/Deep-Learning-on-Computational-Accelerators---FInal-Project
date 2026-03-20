from PIL import Image
import torchvision.transforms.functional as transforms_functional

from .augmentation import Augmentation


class HorizontalFlipAugmentation(Augmentation):
    """Augmentation that mirrors images horizontally."""

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        return transforms_functional.hflip(image)

    def _augmentation_name(self) -> str:
        return "hflip"
