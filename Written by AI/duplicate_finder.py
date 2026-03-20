"""
Duplicate Image Finder using Perceptual Hashing.

Scans a folder for duplicate or near-duplicate images by computing multiple
perceptual hashes (pHash, dHash, aHash, wHash) and grouping images only
when ALL selected algorithms agree they are similar (i.e. the maximum
Hamming distance across all algorithms is within the threshold).

Does not delete or move any files. Saves a timestamped report file in the
scanned folder listing all duplicate groups found.

Usage (command line):
    python duplicate_finder.py <folder_path> [options]

Options:
    --threshold, -t   Max Hamming distance (0 = exact, higher = more lenient).
                      Default: 1
    --algorithms, -a  Which hash algorithms to use. Choose from:
                      phash, dhash, ahash, whash. Default: all four.
    --recursive, -r   Scan subfolders recursively.

Examples:
    python duplicate_finder.py "C:/images" --threshold 2 --recursive
    python duplicate_finder.py "C:/images" -a phash dhash -t 0

Programmatic usage:
    from duplicate_finder import find_duplicates
    groups = find_duplicates("C:/images", distance_threshold=1, recursive=True)

Dependencies:
    pip install Pillow imagehash numpy
"""

import io
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

# Fix Windows console encoding for Unicode filenames
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

try:
    from PIL import Image
    import imagehash
    import numpy as np
except ImportError:
    print("Error: Required packages not found.")
    print("Install them with: pip install Pillow imagehash numpy")
    sys.exit(1)

# Allow very large images (architectural drawings can be huge)
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

AVAILABLE_ALGORITHMS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "ahash": imagehash.average_hash,
    "whash": imagehash.whash,
}

ALL_ALGORITHM_NAMES = list(AVAILABLE_ALGORITHMS.keys())


# ---------------------------------------------------------------------------
#  Image collection
# ---------------------------------------------------------------------------

def collect_image_paths(folder: Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of image file paths found in the folder."""
    files = folder.rglob("*") if recursive else folder.iterdir()
    image_paths = [
        file_path for file_path in files
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


# ---------------------------------------------------------------------------
#  Hashing
# ---------------------------------------------------------------------------

def compute_hashes_for_all_images(
    image_paths: list[Path],
    selected_algorithms: list[str],
) -> tuple[list[Path], dict[str, np.ndarray]]:
    """Compute perceptual hashes for every image using the selected algorithms.

    Returns:
        successfully_hashed_paths: Paths of images that were hashed without error.
        hash_integer_arrays: Dict mapping algorithm name to a numpy uint64 array,
            where element [i] is the hash for successfully_hashed_paths[i].
    """
    selected_hash_functions = {
        name: AVAILABLE_ALGORITHMS[name] for name in selected_algorithms
    }

    successfully_hashed_paths = []
    hash_values_by_algorithm = {name: [] for name in selected_algorithms}
    image_count = len(image_paths)

    for index, image_path in enumerate(image_paths, 1):
        if index % 50 == 0 or index == image_count:
            print(f"  Hashing: {index}/{image_count} images processed...")
        try:
            with Image.open(image_path) as image:
                image_hashes = {
                    algorithm_name: hash_function(image)
                    for algorithm_name, hash_function in selected_hash_functions.items()
                }
            successfully_hashed_paths.append(image_path)
            for algorithm_name in selected_algorithms:
                hash_integer = int(str(image_hashes[algorithm_name]), 16)
                hash_values_by_algorithm[algorithm_name].append(hash_integer)
        except Exception as error:
            print(f"  [WARNING] Could not process {image_path.name}: {error}")

    hash_integer_arrays = {
        algorithm_name: np.array(values, dtype=np.uint64)
        for algorithm_name, values in hash_values_by_algorithm.items()
    }

    return successfully_hashed_paths, hash_integer_arrays


# ---------------------------------------------------------------------------
#  Duplicate detection (Union-Find + pairwise comparison)
# ---------------------------------------------------------------------------

def find_duplicate_groups(
    image_paths: list[Path],
    hash_integer_arrays: dict[str, np.ndarray],
    selected_algorithms: list[str],
    distance_threshold: int = 1,
) -> list[list[Path]]:
    """Group images where ALL selected algorithms agree they are within threshold.

    Uses a Union-Find structure to cluster images. Two images are linked only
    when the maximum Hamming distance across every selected algorithm is at
    or below the threshold.

    Performance notes:
        - Hashes are compared as raw integers via XOR + popcount.
        - An early-exit skips remaining algorithms as soon as one exceeds
          the threshold for a given pair.
    """
    image_count = len(image_paths)

    # Convert numpy arrays to plain Python lists for faster element access
    hash_integers_by_algorithm = {
        algorithm_name: array.tolist()
        for algorithm_name, array in hash_integer_arrays.items()
    }

    # --- Union-Find ---
    parent = list(range(image_count))

    def find_root(node_index):
        while parent[node_index] != node_index:
            parent[node_index] = parent[parent[node_index]]  # path compression
            node_index = parent[node_index]
        return node_index

    def union_nodes(first_node, second_node):
        first_root = find_root(first_node)
        second_root = find_root(second_node)
        if first_root != second_root:
            parent[first_root] = second_root

    # --- Pairwise comparison ---
    total_comparisons = image_count * (image_count - 1) // 2
    algorithm_label = " + ".join(selected_algorithms)
    print(
        f"  Comparing {image_count} images "
        f"({total_comparisons} pairs, threshold={distance_threshold})..."
    )
    print(f"  Mode: all {len(selected_algorithms)} algorithm(s) must agree ({algorithm_label})")

    comparison_count = 0
    for first_index in range(image_count):
        for second_index in range(first_index + 1, image_count):
            is_duplicate = True
            for algorithm_name in selected_algorithms:
                first_hash = hash_integers_by_algorithm[algorithm_name][first_index]
                second_hash = hash_integers_by_algorithm[algorithm_name][second_index]
                hamming_distance = bin(first_hash ^ second_hash).count("1")
                if hamming_distance > distance_threshold:
                    is_duplicate = False
                    break

            if is_duplicate:
                union_nodes(first_index, second_index)

            comparison_count += 1
            if comparison_count % 500_000 == 0:
                print(f"    {comparison_count}/{total_comparisons} comparisons done...")

    # --- Build groups from Union-Find ---
    group_members = defaultdict(list)
    for image_index in range(image_count):
        group_members[find_root(image_index)].append(image_paths[image_index])

    duplicate_groups = [
        members for members in group_members.values() if len(members) > 1
    ]

    for group in duplicate_groups:
        group.sort(key=lambda path: path.name)

    return sorted(duplicate_groups, key=lambda group: group[0].name)


# ---------------------------------------------------------------------------
#  Formatting helpers (shared by console output and saved report)
# ---------------------------------------------------------------------------

def format_group_lines(duplicate_groups: list[list[Path]]) -> list[str]:
    """Return human-readable lines describing each duplicate group."""
    lines = []
    if not duplicate_groups:
        lines.append("No duplicate images found.")
    else:
        for group_number, group in enumerate(duplicate_groups, 1):
            lines.append(f"Group {group_number} ({len(group)} images):")
            for file_path in group:
                file_size_kb = file_path.stat().st_size / 1024
                lines.append(f"  - {file_path.name} ({file_size_kb:.1f} KB)")
            lines.append("")
    return lines


def format_summary_lines(
    total_images_scanned: int,
    duplicate_groups: list[list[Path]],
) -> list[str]:
    """Return human-readable summary statistics lines."""
    total_duplicate_count = sum(len(group) - 1 for group in duplicate_groups)
    unique_count = total_images_scanned - total_duplicate_count
    return [
        f"  Total images scanned:    {total_images_scanned}",
        f"  Duplicate groups found:  {len(duplicate_groups)}",
        f"  Total duplicate images:  {total_duplicate_count}",
        f"  Unique images:           {unique_count}",
    ]


# ---------------------------------------------------------------------------
#  Report saving
# ---------------------------------------------------------------------------

def save_report(
    folder: Path,
    duplicate_groups: list[list[Path]],
    total_images_scanned: int,
    distance_threshold: int,
    selected_algorithms: list[str],
    recursive: bool,
) -> Path:
    """Save a timestamped duplicate report to a text file in the scanned folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = folder / f"duplicate_report_{timestamp}.txt"
    algorithm_label = " + ".join(selected_algorithms)

    lines = [
        "DUPLICATE IMAGE REPORT",
        "=" * 70,
        f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Folder:      {folder}",
        f"Algorithms:  {algorithm_label}",
        f"Threshold:   {distance_threshold} (max Hamming distance across all algorithms)",
        f"Recursive:   {recursive}",
        "=" * 70,
        "",
    ]
    lines.extend(format_group_lines(duplicate_groups))
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.extend(format_summary_lines(total_images_scanned, duplicate_groups))
    lines.append("=" * 70)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def find_duplicates(
    folder_path: str,
    distance_threshold: int = 1,
    algorithms: Optional[list[str]] = None,
    recursive: bool = False,
) -> list[list[Path]]:
    """Scan a folder for duplicate images using multi-algorithm perceptual hashing.

    Args:
        folder_path: Path to the folder to scan.
        distance_threshold: Maximum Hamming distance allowed across all selected
            algorithms for two images to be considered duplicates.
            0 = exact hash match only, higher values = more lenient.
        algorithms: Which hash algorithms to use (any combination of "phash",
            "dhash", "ahash", "whash"). Defaults to all four. Images must
            match on every selected algorithm to be grouped as duplicates.
        recursive: If True, scan subfolders recursively.

    Returns:
        A list of duplicate groups, where each group is a list of Path objects.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return []
    if not folder.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return []

    selected_algorithms = algorithms if algorithms else ALL_ALGORITHM_NAMES

    invalid_algorithms = [
        algorithm for algorithm in selected_algorithms
        if algorithm not in AVAILABLE_ALGORITHMS
    ]
    if invalid_algorithms:
        print(f"Error: Unknown algorithm(s): {invalid_algorithms}")
        print(f"  Available: {ALL_ALGORITHM_NAMES}")
        return []

    algorithm_label = " + ".join(selected_algorithms)

    print("Duplicate Image Finder")
    print("=" * 70)
    print(f"  Folder:      {folder}")
    print(f"  Algorithms:  {algorithm_label} ({len(selected_algorithms)} total)")
    print(f"  Threshold:   {distance_threshold} (max Hamming distance across all algorithms)")
    print(f"  Recursive:   {recursive}")
    print("=" * 70)

    # Step 1: Collect images
    print("\n[1/3] Collecting images...")
    image_paths = collect_image_paths(folder, recursive)
    print(f"  Found {len(image_paths)} images.")

    if len(image_paths) < 2:
        print("  Not enough images to compare.")
        return []

    # Step 2: Compute hashes
    print(f"\n[2/3] Computing perceptual hashes ({len(selected_algorithms)} algorithm(s) per image)...")
    hashed_image_paths, hash_integer_arrays = compute_hashes_for_all_images(
        image_paths, selected_algorithms
    )
    print(f"  Successfully hashed {len(hashed_image_paths)} images.")

    # Step 3: Find duplicates
    print("\n[3/3] Finding duplicates...")
    duplicate_groups = find_duplicate_groups(
        hashed_image_paths, hash_integer_arrays, selected_algorithms, distance_threshold
    )

    # Save report
    report_path = save_report(
        folder, duplicate_groups, len(hashed_image_paths),
        distance_threshold, selected_algorithms, recursive
    )

    # Print results to console (uses the same formatting as the saved report)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for line in format_group_lines(duplicate_groups):
        print(line)
    print("=" * 70)
    print("Summary:")
    for line in format_summary_lines(len(hashed_image_paths), duplicate_groups):
        print(line)
    print(f"\n  Report saved to: {report_path}")
    print("=" * 70)

    return duplicate_groups


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate images using multi-algorithm perceptual hashing."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder to scan for duplicate images.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=1,
        help="Max Hamming distance across ALL selected algorithms. "
             "0 = exact match, higher = more lenient. Default: 1",
    )
    parser.add_argument(
        "--algorithms", "-a",
        nargs="+",
        choices=ALL_ALGORITHM_NAMES,
        default=ALL_ALGORITHM_NAMES,
        help="Hash algorithms to use (default: all four). "
             "Images must match on ALL selected algorithms.",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan subfolders recursively.",
    )

    args = parser.parse_args()

    find_duplicates(
        folder_path=args.folder,
        distance_threshold=args.threshold,
        algorithms=args.algorithms,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
