import random

from PIL import Image
import torchvision.transforms.functional as transforms_functional

from .augmentation import Augmentation


class ColorJitterAugmentation(Augmentation):
    """Augmentation that randomly adjusts brightness, contrast, and saturation."""

    def __init__(
        self,
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
        saturation_jitter: float = 0.2,
    ):
        """
        :param brightness_jitter: Maximum fractional change in brightness.
            A value of 0.2 means brightness is scaled by a factor in [0.8, 1.2].
        :param contrast_jitter: Maximum fractional change in contrast.
        :param saturation_jitter: Maximum fractional change in saturation.
        """
        self._brightness_jitter = brightness_jitter
        self._contrast_jitter = contrast_jitter
        self._saturation_jitter = saturation_jitter

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        brightness_factor = random.uniform(1.0 - self._brightness_jitter, 1.0 + self._brightness_jitter)
        contrast_factor = random.uniform(1.0 - self._contrast_jitter, 1.0 + self._contrast_jitter)
        saturation_factor = random.uniform(1.0 - self._saturation_jitter, 1.0 + self._saturation_jitter)

        augmented_image = transforms_functional.adjust_brightness(image, brightness_factor)
        augmented_image = transforms_functional.adjust_contrast(augmented_image, contrast_factor)
        augmented_image = transforms_functional.adjust_saturation(augmented_image, saturation_factor)
        return augmented_image

    def _augmentation_name(self) -> str:
        return "colorjitter"
