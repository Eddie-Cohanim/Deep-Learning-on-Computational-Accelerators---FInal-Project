"""
Run this script from inside the Augmentations folder to visually inspect every augmentation.

Usage:
    python test_augmentations.py --image "path/to/your/image.png"

Results are saved to a tmp_augmentations/ folder created next to this script.
"""

import argparse
import pathlib

from PIL import Image

from augmentations import DEFAULT_AUGMENTATIONS

AUGMENTATIONS = DEFAULT_AUGMENTATIONS

def main():
    parser = argparse.ArgumentParser(description="Visually test all augmentations on a single image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path} ({original_image.width}x{original_image.height})")
    print()

    tmp_dir = pathlib.Path(__file__).parent / "tmp_augmentations"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {tmp_dir.resolve()}")
    print()

    for augmentation in AUGMENTATIONS:
        name = augmentation._augmentation_name()
        augmented_image = augmentation._apply_to_image(original_image)

        output_path = tmp_dir / f"{image_path.stem}_{name}{image_path.suffix}"
        augmented_image.save(output_path)
        print(f"  {name:20s} -> {output_path.name}")

    print()
    print("Done. Open tmp_augmentations/ in VS Code to preview the results.")


if __name__ == "__main__":
    main()