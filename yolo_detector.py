"""
YOLO Object Detection for Dataset Images.

Scans a folder of images, runs a YOLO model to detect objects, and saves the
bounding box results to a JSON file. The JSON output can then be consumed by
a separate cropping script to extract detected regions.

Usage:
    python yolo_detector.py <folder_path> [--model yolo11n.pt] [--confidence 0.25]
                                          [--target-classes cup "wine glass"]
                                          [--recursive]

Dependencies:
    pip install ultralytics
"""

from __future__ import annotations

import io
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for Unicode filenames
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Required package not found.")
    print("Install it with: pip install ultralytics")
    sys.exit(1)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

# Default class filter — covers the COCO names used by YOLO11 as well as
# common variations in case a custom model is used.
DEFAULT_TARGET_CLASS_NAMES = ["cup", "wine glass", "glass", "mug", "beer", "glass of wine"]


def collect_image_paths(folder: Path, recursive: bool = False) -> list[Path]:
    """Collect all image file paths from the folder."""
    if recursive:
        files = folder.rglob("*")
    else:
        files = folder.iterdir()

    image_paths = [
        file_path for file_path in files
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


class YOLODetector:
    """
    Runs YOLO object detection on images and collects bounding box results.

    Loads a pre-trained YOLO model and provides methods to detect objects in
    individual images or entire folders. Detection results are returned as
    dictionaries mapping image filenames to lists of detected objects, each
    with a class name, confidence score, and bounding box coordinates.

    Results can be saved to a JSON file for downstream processing such as
    cropping detected regions from the original images.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        target_class_names: list[str] | None = None,
    ):
        """
        :param model_name: Name or path of the YOLO model weights file.
            If the file does not exist locally, ultralytics will attempt to
            download it automatically.
        :param confidence_threshold: Minimum confidence score for a detection
            to be included in the results. Detections below this threshold
            are discarded.
        :param target_class_names: Optional list of class names to keep.
            When provided, only detections whose class name appears in this
            list are included. When None, all detected classes are kept.
        """
        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold
        self._target_class_names = (
            set(name.lower() for name in target_class_names)
            if target_class_names is not None
            else None
        )

    def detect_objects_in_image(self, image_path: Path) -> list[dict]:
        """
        Runs YOLO detection on a single image and returns the results.

        :param image_path: Path to the image file to process.
        :return: A list of detection dictionaries, each containing:
            - "class_name": the detected object's class label (str)
            - "confidence": the detection confidence score (float)
            - "bounding_box": a dict with "x_min", "y_min", "x_max", "y_max"
              pixel coordinates (int)
        """
        prediction_results = self._model.predict(
            source=str(image_path),
            conf=self._confidence_threshold,
            verbose=False,
        )

        detections = []
        for result in prediction_results:
            class_name_mapping = result.names
            boxes = result.boxes

            for box_index in range(len(boxes)):
                class_id = int(boxes.cls[box_index].item())
                class_name = class_name_mapping[class_id]

                if self._target_class_names is not None:
                    if class_name.lower() not in self._target_class_names:
                        continue

                confidence = round(boxes.conf[box_index].item(), 4)
                x_min, y_min, x_max, y_max = boxes.xyxy[box_index].tolist()

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": {
                        "x_min": int(round(x_min)),
                        "y_min": int(round(y_min)),
                        "x_max": int(round(x_max)),
                        "y_max": int(round(y_max)),
                    },
                })

        return detections

    def detect_objects_in_folder(
        self,
        folder_path: Path,
        recursive: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Runs YOLO detection on all images in a folder.

        :param folder_path: Root folder containing images to process.
        :param recursive: Whether to scan subfolders recursively.
        :return: A dictionary mapping each image's path (relative to
            folder_path) to its list of detection results.
        """
        image_paths = collect_image_paths(folder_path, recursive)
        total_image_count = len(image_paths)
        detection_results = {}

        for image_index, image_path in enumerate(image_paths, start=1):
            if image_index % 10 == 0 or image_index == total_image_count:
                print(f"  Processing: {image_index}/{total_image_count} images...")

            try:
                detections = self.detect_objects_in_image(image_path)
                relative_path = str(image_path.relative_to(folder_path))
                detection_results[relative_path] = detections
            except Exception as error:
                print(f"  [WARNING] Could not process {image_path.name}: {error}")

        return detection_results

    @staticmethod
    def save_detection_results(
        detection_results: dict[str, list[dict]],
        output_path: Path,
    ) -> Path:
        """
        Saves the detection results dictionary to a JSON file.

        :param detection_results: The detection results as returned by
            detect_objects_in_folder.
        :param output_path: Path where the JSON file will be written.
        :return: The path the file was written to.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(detection_results, output_file, indent=2, ensure_ascii=False)
        return output_path


def run_detection(
    folder_path: str,
    model_name: str = "yolo11n.pt",
    confidence_threshold: float = 0.25,
    target_class_names: list[str] = DEFAULT_TARGET_CLASS_NAMES,
    recursive: bool = False,
) -> dict[str, list[dict]]:
    """
    Main entry point for running YOLO detection on a folder of images.

    :param folder_path: Path to the folder containing images to scan.
    :param model_name: YOLO model weights name or path.
    :param confidence_threshold: Minimum detection confidence to include.
    :param target_class_names: Class names to keep. Defaults to
        DEFAULT_TARGET_CLASS_NAMES (cup, wine glass, glass, mug, beer, glass of wine).
    :param recursive: Whether to scan subfolders recursively.
    :return: Detection results dictionary.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return {}
    if not folder.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return {}

    print("YOLO Object Detection")
    print("=" * 70)
    print(f"  Folder:      {folder}")
    print(f"  Model:       {model_name}")
    print(f"  Confidence:  {confidence_threshold}")
    print(f"  Classes:     {target_class_names or 'all'}")
    print(f"  Recursive:   {recursive}")
    print("=" * 70)

    # Step 1: Collect images
    print("\n[1/3] Collecting images...")
    image_paths = collect_image_paths(folder, recursive)
    print(f"  Found {len(image_paths)} images.")

    if not image_paths:
        print("  No images found in the specified folder.")
        return {}

    # Step 2: Run detection
    print("\n[2/3] Running YOLO detection...")
    detector = YOLODetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        target_class_names=target_class_names,
    )
    detection_results = detector.detect_objects_in_folder(folder, recursive)

    # Step 3: Save results
    print("\n[3/3] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = folder / f"yolo_detections_{timestamp}.json"
    detector.save_detection_results(detection_results, output_path)

    # Print summary
    total_detections = sum(len(detections) for detections in detection_results.values())
    images_with_detections = sum(
        1 for detections in detection_results.values() if detections
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total images processed:       {len(detection_results)}")
    print(f"  Images with detections:       {images_with_detections}")
    print(f"  Images without detections:    {len(detection_results) - images_with_detections}")
    print(f"  Total detections:             {total_detections}")
    print(f"\n  Results saved to: {output_path}")
    print("=" * 70)

    return detection_results


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on a folder of images and save results to JSON."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing images to scan.",
    )
    parser.add_argument(
        "--model", "-m",
        default="yolo11n.pt",
        help="YOLO model name or path to weights file. Default: yolo11n.pt",
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for detections (0.0 to 1.0). Default: 0.25",
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=DEFAULT_TARGET_CLASS_NAMES,
        help=(
            "Filter detections to specific class names. "
            f"Default: {DEFAULT_TARGET_CLASS_NAMES}"
        ),
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan subfolders recursively.",
    )

    args = parser.parse_args()

    run_detection(
        folder_path=args.folder,
        model_name=args.model,
        confidence_threshold=args.confidence,
        target_class_names=args.target_classes,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
