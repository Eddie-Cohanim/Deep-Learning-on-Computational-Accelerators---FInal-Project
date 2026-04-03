import random

from PIL import Image, ImageFilter

from .augmentation import Augmentation


class GaussianBlurAugmentation(Augmentation):
    """Augmentation that applies a Gaussian blur with a randomly sampled radius."""

    def __init__(
        self,
        minimum_blur_radius: float = 0.5,
        maximum_blur_radius: float = 2.0,
        augmentation_fraction: float = 0.5,
    ) -> None:
        """
        :param minimum_blur_radius: Lower bound of the blur radius range.
        :param maximum_blur_radius: Upper bound of the blur radius range.
        :param augmentation_fraction: Fraction of eligible images to augment.
        """
        super().__init__(augmentation_fraction=augmentation_fraction)
        self._minimum_blur_radius = minimum_blur_radius
        self._maximum_blur_radius = maximum_blur_radius

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        blur_radius = random.uniform(self._minimum_blur_radius, self._maximum_blur_radius)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _augmentation_name(self) -> str:
        return "gaussianblur"
