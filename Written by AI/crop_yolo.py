from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def sanitize_name(name: str) -> str:
    """
    Convert a class name into a safe folder/file name.

    Example:
        'wine glass' -> 'wine_glass'
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-]", "", name)
    return name or "unknown"


def clip_box_to_image(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """
    Convert box coordinates to valid integer pixel coordinates and clip them
    to the image boundaries.

    Returns:
        (x1, y1, x2, y2) if valid
        None if the box becomes invalid after clipping
    """
    values = [x_min, y_min, x_max, y_max]
    if not np.all(np.isfinite(values)):
        return None

    # Handle reversed coordinates safely
    x1, x2 = sorted((x_min, x_max))
    y1, y2 = sorted((y_min, y_max))

    # Convert float coordinates to integer pixel indices
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))

    # Clip to image boundaries
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    # Invalid or empty crop
    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_and_save_by_class(
    input_folder: str | Path,
    detections_json_path: str | Path,
    confidence_threshold: float = 0.0,
) -> Path:
    """
    Read images from input_folder and detections from a JSON file, then:
    - create <input_folder>_croped
    - create one subfolder per class
    - save each crop into the matching class folder

    Expected JSON structure:
    {
        "image1.png": [
            {
                "class_name": "bottle",
                "confidence": 0.91,
                "bounding_box": {
                    "x_min": 10,
                    "y_min": 20,
                    "x_max": 100,
                    "y_max": 200
                }
            }
        ]
    }
    """
    input_path = Path(input_folder)
    json_path = Path(detections_json_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"Detections JSON file does not exist: {json_path}")

    output_root = input_path.parent / f"{input_path.name}_croped"
    output_root.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        detections_by_image: Dict[str, List[Dict[str, Any]]] = json.load(f)

    for image_name, detections in detections_by_image.items():
        image_path = input_path / image_name

        if not image_path.exists():
            print(f"Skipping missing image: {image_name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_name}")
            continue

        image_height, image_width = image.shape[:2]

        # Counter per class for this image, so filenames stay unique
        class_counters: Dict[str, int] = {}

        for detection in detections:
            class_name = detection.get("class_name")
            confidence = detection.get("confidence", 0.0)
            bounding_box = detection.get("bounding_box", {})

            if not class_name:
                continue

            if confidence < confidence_threshold:
                continue

            x_min = bounding_box.get("x_min")
            y_min = bounding_box.get("y_min")
            x_max = bounding_box.get("x_max")
            y_max = bounding_box.get("y_max")

            if None in (x_min, y_min, x_max, y_max):
                continue

            clipped_box = clip_box_to_image(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                image_width=image_width,
                image_height=image_height,
            )

            if clipped_box is None:
                continue

            x1, y1, x2, y2 = clipped_box
            crop = image[y1:y2, x1:x2].copy()

            safe_class_name = sanitize_name(class_name)
            class_folder = output_root / safe_class_name
            class_folder.mkdir(parents=True, exist_ok=True)

            class_counters[safe_class_name] = class_counters.get(safe_class_name, 0) + 1
            instance_index = class_counters[safe_class_name]

            output_filename = (
                f"{Path(image_name).stem}_{safe_class_name}_{instance_index}"
                f"{Path(image_name).suffix}"
            )
            output_path = class_folder / output_filename

            success = cv2.imwrite(str(output_path), crop)
            if success:
                print(f"Saved: {output_path}")
            else:
                print(f"Failed to save: {output_path}")

    return output_root


def main() -> None:
    input_folder = "rose wine"
    detections_json_path = "yolo_detections_20260325_161437.json"

    output_folder = crop_and_save_by_class(
        input_folder=input_folder,
        detections_json_path=detections_json_path,
        confidence_threshold=0.25,  # optional, change if needed
    )

    print(f"\nDone. Output saved to: {output_folder}")


if __name__ == "__main__":
    main()