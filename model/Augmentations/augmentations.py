import abc
import pathlib
import random

import numpy as np
from PIL import Image, ImageFilter


# NOTE: A global list of augmented files still needs to be implemented in the main pipeline.
# This list should be initialized once (e.g. as an empty list) before any augmentation
# is performed, then passed to each augment() call. The method updates it in-place so
# that no file is ever augmented twice across multiple augmentation passes.

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

        number_of_files_to_augment = random.randint(1, len(eligible_image_paths))
        selected_image_paths = random.sample(eligible_image_paths, number_of_files_to_augment)

        for original_image_path in selected_image_paths:
            original_image = Image.open(original_image_path).convert("RGB")
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


# ---------------------------------------------------------------------------
# Geometric augmentations
# ---------------------------------------------------------------------------

class HorizontalFlipAugmentation(Augmentation):
    """
    Mirrors images horizontally by reversing the column order of each row.

    Equivalent to reflecting the image across its vertical centre axis.
    A glass of wine photographed from the left side is visually identical
    to one photographed from the right — making this a label-safe augmentation.
    """

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        pixels = np.array(image)
        flipped_pixels = pixels[:, ::-1, :]
        return Image.fromarray(flipped_pixels.astype(np.uint8))

    def _augmentation_name(self) -> str:
        return "hflip"


class RotationAugmentation(Augmentation):
    """
    Rotates the image by a random angle using an affine transformation matrix.

    The rotation is performed around the image centre. Pixels that fall outside
    the original canvas after rotation are filled with black (0, 0, 0). A small
    angle range (default ±15°) keeps glasses upright and realistic.

    Implementation: for each output pixel we compute its corresponding source
    coordinates using the inverse rotation matrix, then sample the source image
    with nearest-neighbour interpolation.
    """

    def __init__(self, max_rotation_degrees: float = 15.0):
        """
        :param max_rotation_degrees: Maximum rotation magnitude in degrees.
            The actual angle is sampled uniformly from
            [-max_rotation_degrees, +max_rotation_degrees].
        """
        self._max_rotation_degrees = max_rotation_degrees

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        angle_degrees = random.uniform(-self._max_rotation_degrees, self._max_rotation_degrees)
        return image.rotate(angle_degrees, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)

    def _augmentation_name(self) -> str:
        return "rotation"


class PerspectiveAugmentation(Augmentation):
    """
    Applies a random perspective warp to simulate different camera angles.

    A perspective transform maps four source corner points to four slightly
    displaced destination points, producing the effect of viewing the image
    from a different angle. This simulates overhead shots, angled bar photos,
    and varying distances — all common in open-source beverage image sets.

    Implementation: we compute the 3×3 homography matrix from the four point
    correspondences and apply it as an inverse warp (destination → source)
    with nearest-neighbour sampling.
    """

    def __init__(self, distortion_scale: float = 0.2):
        """
        :param distortion_scale: Controls how far each corner can be displaced,
            as a fraction of the shorter image dimension. Keep ≤ 0.3 to avoid
            unrealistic warps.
        """
        self._distortion_scale = distortion_scale

    @staticmethod
    def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Solves for the 3×3 homography H such that dst ≈ H @ src (in homogeneous coords).
        Uses the direct linear transform (DLT) with 4 point correspondences.
        """
        A = []
        for (x, y), (x_prime, y_prime) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime])
            A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
        A = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_displacement = self._distortion_scale * min(width, height) / 2.0

        def rand_offset() -> float:
            return random.uniform(-max_displacement, max_displacement)

        src_corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float64,
        )
        dst_corners = src_corners + np.array([
            [rand_offset(), rand_offset()],
            [rand_offset(), rand_offset()],
            [rand_offset(), rand_offset()],
            [rand_offset(), rand_offset()],
        ])

        homography_matrix = self._compute_homography(dst_corners, src_corners)
        homography_matrix = homography_matrix / homography_matrix[2, 2]
        pil_coefficients = [
            homography_matrix[0, 0], homography_matrix[0, 1], homography_matrix[0, 2],
            homography_matrix[1, 0], homography_matrix[1, 1], homography_matrix[1, 2],
            homography_matrix[2, 0], homography_matrix[2, 1],
        ]
        return image.transform(
            (width, height),
            Image.PERSPECTIVE,
            pil_coefficients,
            resample=Image.Resampling.BILINEAR,
        )

    def _augmentation_name(self) -> str:
        return "perspective"


# ---------------------------------------------------------------------------
# Color / appearance augmentations
# ---------------------------------------------------------------------------

class ColorJitterAugmentation(Augmentation):
    """
    Randomly adjusts brightness, contrast, and saturation.

    All three operations are implemented directly on the pixel array:
    - Brightness: scale all channels by a constant factor.
    - Contrast:   interpolate each channel toward its mean grey value.
    - Saturation: interpolate each pixel toward its luminance (greyscale) value.

    Values are clipped to [0, 255] after each operation.
    """

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
        pixels = np.array(image, dtype=np.float32)

        brightness_factor = random.uniform(
            1.0 - self._brightness_jitter, 1.0 + self._brightness_jitter
        )
        pixels = pixels * brightness_factor

        contrast_factor = random.uniform(
            1.0 - self._contrast_jitter, 1.0 + self._contrast_jitter
        )
        channel_means = pixels.mean(axis=(0, 1), keepdims=True)
        pixels = contrast_factor * pixels + (1.0 - contrast_factor) * channel_means

        saturation_factor = random.uniform(
            1.0 - self._saturation_jitter, 1.0 + self._saturation_jitter
        )
        luminance = (
            0.299 * pixels[:, :, 0:1] +
            0.587 * pixels[:, :, 1:2] +
            0.114 * pixels[:, :, 2:3]
        )
        pixels = saturation_factor * pixels + (1.0 - saturation_factor) * luminance

        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    def _augmentation_name(self) -> str:
        return "colorjitter"




class GammaAugmentation(Augmentation):
    """
    Applies random gamma correction to simulate different camera exposures.

    Gamma correction is a non-linear operation: output = (input / 255) ^ gamma * 255.
    Gamma < 1 brightens mid-tones (over-exposed look); gamma > 1 darkens them
    (under-exposed, dimly-lit bar). Unlike brightness scaling, gamma affects
    highlights and shadows differently, which better matches real camera variation.
    """

    def __init__(self, gamma_range: tuple[float, float] = (0.7, 1.4)):
        """
        :param gamma_range: (min_gamma, max_gamma) for the uniform sample.
        """
        self._gamma_min, self._gamma_max = gamma_range

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        gamma = random.uniform(self._gamma_min, self._gamma_max)
        pixels = np.array(image, dtype=np.float32) / 255.0
        corrected = np.power(pixels, gamma)
        return Image.fromarray(np.clip(corrected * 255, 0, 255).astype(np.uint8))

    def _augmentation_name(self) -> str:
        return "gamma"


# ---------------------------------------------------------------------------
# Noise / quality augmentations
# ---------------------------------------------------------------------------

class GaussianBlurAugmentation(Augmentation):
    """
    Applies a Gaussian blur using a manually constructed convolution kernel.

    Beverage photos taken on phones are often slightly out-of-focus, especially
    close-up shots of foam or the glass surface. Training with blurred samples
    prevents the model from over-relying on fine textures absent at inference time.

    Implementation: build a 1D Gaussian kernel, then convolve each RGB channel
    independently using a separable pass (row-wise then column-wise) for efficiency.
    This is mathematically identical to a 2D Gaussian convolution.
    """

    def __init__(self, max_sigma: float = 1.5):
        """
        :param max_sigma: Maximum standard deviation of the Gaussian kernel in pixels.
            Values above 2.0 obscure colour information too aggressively.
        """
        self._max_sigma = max_sigma

    def _apply_to_image(self, image: Image.Image) -> Image.Image:
        sigma = random.uniform(0.3, self._max_sigma)
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _augmentation_name(self) -> str:
        return "gblur"
    

DEFAULT_AUGMENTATIONS = [
    HorizontalFlipAugmentation(),
    RotationAugmentation(max_rotation_degrees=15.0),
    PerspectiveAugmentation(distortion_scale=0.2),
    ColorJitterAugmentation(brightness_jitter=0.3, contrast_jitter=0.3, saturation_jitter=0.3),
    GammaAugmentation(gamma_range=(0.7, 1.4)),
    GaussianBlurAugmentation(max_sigma=1.5),
]
