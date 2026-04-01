"""
YOLO Detection Visualizer.

Reads a JSON detections file produced by yolo_detector.py and draws bounding
boxes with class labels and confidence scores onto the original images. Saves
the annotated images to an output folder without modifying the originals.

Usage:
    python yolo_visualizer.py <detections_json> [--output-folder ./visualized]
                                                [--box-thickness 3]
                                                [--font-size 16]

Dependencies:
    pip install Pillow
"""

from __future__ import annotations

import io
import sys
import json
import argparse
from pathlib import Path

# Fix Windows console encoding for Unicode filenames
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Required package not found.")
    print("Install it with: pip install Pillow")
    sys.exit(1)


# One distinct color per class name, cycling through this palette
_BOUNDING_BOX_COLOR_PALETTE = [
    (255,  87,  87),  # red
    ( 87, 183, 255),  # blue
    ( 87, 255, 148),  # green
    (255, 214,  87),  # yellow
    (214,  87, 255),  # purple
    (255, 148,  87),  # orange
    ( 87, 255, 255),  # cyan
    (255,  87, 214),  # pink
]


class DetectionVisualizer:
    """
    Draws YOLO bounding box detections onto images and saves the results.

    Reads detection results from a JSON file (as produced by yolo_detector.py)
    and annotates each original image with labeled bounding boxes. Each class
    name is assigned a consistent color across all images. Annotated images are
    saved to a specified output folder, preserving the original files.
    """

    def __init__(
        self,
        box_thickness: int = 3,
        font_size: int = 16,
    ):
        """
        :param box_thickness: Pixel width of the drawn bounding box borders.
        :param font_size: Font size for the class label text drawn above each box.
        """
        self._box_thickness = box_thickness
        self._font_size = font_size
        self._class_name_to_color: dict[str, tuple[int, int, int]] = {}
        self._font = self._load_font(font_size)

    @staticmethod
    def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """
        Attempts to load a TrueType font at the given size.
        Falls back to the PIL default bitmap font if no TrueType font is found.
        """
        truetype_font_names = [
            "arial.ttf",
            "Arial.ttf",
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
        ]
        for font_name in truetype_font_names:
            try:
                return ImageFont.truetype(font_name, size=font_size)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    def _get_color_for_class(self, class_name: str) -> tuple[int, int, int]:
        """
        Returns a consistent color for the given class name, assigning a new
        one from the palette if this class has not been seen before.
        """
        if class_name not in self._class_name_to_color:
            color_index = len(self._class_name_to_color) % len(_BOUNDING_BOX_COLOR_PALETTE)
            self._class_name_to_color[class_name] = _BOUNDING_BOX_COLOR_PALETTE[color_index]
        return self._class_name_to_color[class_name]

    def visualize_image(
        self,
        image_path: Path,
        detections: list[dict],
    ) -> Image.Image:
        """
        Draws bounding boxes and labels onto a single image.

        :param image_path: Path to the original image file.
        :param detections: List of detection dicts, each with keys
            "class_name", "confidence", and "bounding_box"
            (containing "x_min", "y_min", "x_max", "y_max").
        :return: A new PIL image with all detections annotated.
        """
        original_image = Image.open(image_path).convert("RGB")
        annotated_image = original_image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            bounding_box = detection["bounding_box"]

            x_min = bounding_box["x_min"]
            y_min = bounding_box["y_min"]
            x_max = bounding_box["x_max"]
            y_max = bounding_box["y_max"]

            box_color = self._get_color_for_class(class_name)
            label_text = f"{class_name} {confidence:.2f}"

            # Draw bounding box
            for thickness_offset in range(self._box_thickness):
                draw.rectangle(
                    [
                        x_min - thickness_offset,
                        y_min - thickness_offset,
                        x_max + thickness_offset,
                        y_max + thickness_offset,
                    ],
                    outline=box_color,
                )

            # Measure label text to size the background rectangle
            text_bounding_box = draw.textbbox((x_min, y_min), label_text, font=self._font)
            text_width = text_bounding_box[2] - text_bounding_box[0]
            text_height = text_bounding_box[3] - text_bounding_box[1]
            label_padding = 3

            label_background_top = y_min - text_height - label_padding * 2
            label_background_bottom = y_min

            # Keep label inside image bounds if box is near the top
            if label_background_top < 0:
                label_background_top = y_min
                label_background_bottom = y_min + text_height + label_padding * 2

            draw.rectangle(
                [
                    x_min,
                    label_background_top,
                    x_min + text_width + label_padding * 2,
                    label_background_bottom,
                ],
                fill=box_color,
            )
            draw.text(
                (x_min + label_padding, label_background_top + label_padding),
                label_text,
                fill=(255, 255, 255),
                font=self._font,
            )

        return annotated_image

    def visualize_folder(
        self,
        image_folder: Path,
        detection_results: dict[str, list[dict]],
        output_folder: Path,
    ) -> int:
        """
        Annotates all images referenced in the detection results and saves them.

        :param image_folder: Root folder of the original images, used to resolve
            relative paths stored in the detection results JSON.
        :param detection_results: The parsed detections dictionary mapping
            relative image paths to their detection lists.
        :param output_folder: Folder where annotated images will be saved.
            Created automatically if it does not exist. The subfolder structure
            of the original images is preserved.
        :return: The number of images successfully annotated and saved.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        successfully_saved_count = 0
        total_image_count = len(detection_results)

        for image_index, (relative_image_path, detections) in enumerate(
            detection_results.items(), start=1
        ):
            if image_index % 10 == 0 or image_index == total_image_count:
                print(f"  Annotating: {image_index}/{total_image_count} images...")

            original_image_path = image_folder / relative_image_path

            if not original_image_path.exists():
                print(f"  [WARNING] Image not found, skipping: {original_image_path}")
                continue

            try:
                annotated_image = self.visualize_image(original_image_path, detections)

                output_image_path = output_folder / relative_image_path
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                annotated_image.save(output_image_path)
                successfully_saved_count += 1
            except Exception as error:
                print(f"  [WARNING] Could not annotate {original_image_path.name}: {error}")

        return successfully_saved_count


def run_visualization(
    detections_json_path: str,
    output_folder_path: str | None = None,
    box_thickness: int = 3,
    font_size: int = 16,
) -> int:
    """
    Main entry point for visualizing YOLO detections.

    :param detections_json_path: Path to the JSON detections file produced
        by yolo_detector.py.
    :param output_folder_path: Folder to save annotated images. Defaults to
        a 'visualized' subfolder next to the detections JSON file.
    :param box_thickness: Pixel width of the bounding box borders.
    :param font_size: Font size for class label text.
    :return: Number of images successfully annotated.
    """
    detections_json = Path(detections_json_path)
    if not detections_json.exists():
        print(f"Error: Detections file does not exist: {detections_json_path}")
        return 0
    if not detections_json.is_file():
        print(f"Error: Path is not a file: {detections_json_path}")
        return 0

    image_folder = detections_json.parent

    if output_folder_path is not None:
        output_folder = Path(output_folder_path)
    else:
        output_folder = image_folder / "visualized"

    print("YOLO Detection Visualizer")
    print("=" * 70)
    print(f"  Detections file:  {detections_json}")
    print(f"  Image folder:     {image_folder}")
    print(f"  Output folder:    {output_folder}")
    print(f"  Box thickness:    {box_thickness}px")
    print(f"  Font size:        {font_size}pt")
    print("=" * 70)

    # Step 1: Load detections
    print("\n[1/3] Loading detections...")
    with open(detections_json, "r", encoding="utf-8") as detections_file:
        detection_results = json.load(detections_file)

    total_detections = sum(len(detections) for detections in detection_results.values())
    print(f"  Loaded {len(detection_results)} images with {total_detections} total detections.")

    # Step 2: Annotate images
    print("\n[2/3] Annotating images...")
    visualizer = DetectionVisualizer(
        box_thickness=box_thickness,
        font_size=font_size,
    )
    successfully_saved_count = visualizer.visualize_folder(
        image_folder=image_folder,
        detection_results=detection_results,
        output_folder=output_folder,
    )

    # Step 3: Summary
    print("\n[3/3] Done.")
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Images annotated:  {successfully_saved_count} / {len(detection_results)}")
    print(f"  Output folder:     {output_folder}")
    print("=" * 70)

    return successfully_saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO detections by drawing bounding boxes on images."
    )
    parser.add_argument(
        "detections_json",
        help="Path to the JSON detections file produced by yolo_detector.py.",
    )
    parser.add_argument(
        "--output-folder", "-o",
        default=None,
        help="Folder to save annotated images. Defaults to a 'visualized' subfolder next to the JSON file.",
    )
    parser.add_argument(
        "--box-thickness", "-t",
        type=int,
        default=3,
        help="Pixel width of the bounding box borders. Default: 3",
    )
    parser.add_argument(
        "--font-size", "-f",
        type=int,
        default=16,
        help="Font size for class label text. Default: 16",
    )

    args = parser.parse_args()

    run_visualization(
        detections_json_path=args.detections_json,
        output_folder_path=args.output_folder,
        box_thickness=args.box_thickness,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
