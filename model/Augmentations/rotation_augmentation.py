import random

from PIL import Image
import torchvision.transforms.functional as transforms_functional

from .augmentation import Augmentation


class RotationAugmentation(Augmentation):
    """Augmentation that rotates images by a random angle within a configured range."""

    def __init__(self, max_rotation_degrees: float = 15.0, augmentation_fraction: float = 0.5) -> None:
        """
        :param max_rotation_degrees: Maximum rotation magnitude in degrees.
            The actual angle is sampled uniformly from
            [-max_rotation_degrees, +max_rotation_degrees].
        :param augmentation_fraction: Fraction of eligible images to augment.
        """
        super().__init__(augmentation_fraction=augmentation_fraction)
        self._max_rotation_degrees = max_rotation_degrees

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        rotation_angle = random.uniform(-self._max_rotation_degrees, self._max_rotation_degrees)
        return transforms_functional.rotate(image, rotation_angle)

    def _augmentation_name(self) -> str:
        return "rotation"
