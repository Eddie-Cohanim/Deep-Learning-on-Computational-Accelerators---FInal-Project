"""
Duplicate Image Finder using Perceptual Hashing.

Scans a folder for duplicate/near-duplicate images by computing perceptual
hashes (pHash, dHash, aHash) and grouping images whose hashes fall within
a configurable Hamming distance threshold.

Saves a report file in the scanned folder with the list of duplicates found.

Usage:
    python duplicate_finder.py <folder_path> [--threshold 6] [--algorithm phash]
                                              [--recursive]

Dependencies:
    pip install Pillow imagehash
"""

import io
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Fix Windows console encoding for Unicode filenames
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Error: Required packages not found.")
    print("Install them with: pip install Pillow imagehash")
    sys.exit(1)

# Allow very large images (architectural drawings can be huge)
Image.MAX_IMAGE_PIXELS = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

HASH_FUNCTIONS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "ahash": imagehash.average_hash,
    "whash": imagehash.whash,
}


def collect_image_paths(folder: Path, recursive: bool = False) -> list[Path]:
    """Collect all image file paths from the folder."""
    if recursive:
        files = folder.rglob("*")
    else:
        files = folder.iterdir()

    image_paths = [
        f for f in files
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def compute_hashes(
    image_paths: list[Path],
    algorithm: str = "phash",
) -> dict[Path, imagehash.ImageHash]:
    """Compute perceptual hash for each image. Returns {path: hash}."""
    hash_fn = HASH_FUNCTIONS[algorithm]
    hashes = {}
    total = len(image_paths)

    for i, path in enumerate(image_paths, 1):
        if i % 50 == 0 or i == total:
            print(f"  Hashing: {i}/{total} images processed...")
        try:
            with Image.open(path) as img:
                hashes[path] = hash_fn(img)
        except Exception as e:
            print(f"  [WARNING] Could not process {path.name}: {e}")

    return hashes


def find_duplicate_groups(
    hashes: dict[Path, imagehash.ImageHash],
    threshold: int = 6,
) -> list[list[Path]]:
    """Group images whose perceptual hashes are within the Hamming distance threshold.

    Uses Union-Find to cluster images so that transitive near-duplicates are
    grouped together.
    """
    paths = list(hashes.keys())
    n = len(paths)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    print(f"  Comparing {n} images (threshold={threshold})...")
    comparisons = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = hashes[paths[i]] - hashes[paths[j]]
            if distance <= threshold:
                union(i, j)
            comparisons += 1
            if comparisons % 500_000 == 0:
                print(f"    {comparisons} comparisons done...")

    # Build groups
    groups_map = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(paths[i])

    # Only return groups with more than one image (actual duplicates)
    duplicate_groups = [g for g in groups_map.values() if len(g) > 1]

    # Sort each group by filename for consistent output
    for g in duplicate_groups:
        g.sort(key=lambda p: p.name)

    return sorted(duplicate_groups, key=lambda g: g[0].name)


def save_report(
    folder: Path,
    duplicate_groups: list[list[Path]],
    total_images: int,
    algorithm: str,
    threshold: int,
    recursive: bool,
) -> Path:
    """Save the duplicate report to a text file in the scanned folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = folder / f"duplicate_report_{timestamp}.txt"

    total_duplicates = sum(len(g) - 1 for g in duplicate_groups)

    lines = []
    lines.append("DUPLICATE IMAGE REPORT")
    lines.append("=" * 70)
    lines.append(f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Folder:      {folder}")
    lines.append(f"Algorithm:   {algorithm}")
    lines.append(f"Threshold:   {threshold}")
    lines.append(f"Recursive:   {recursive}")
    lines.append("=" * 70)
    lines.append("")

    if not duplicate_groups:
        lines.append("No duplicate images found.")
    else:
        for idx, group in enumerate(duplicate_groups, 1):
            lines.append(f"Group {idx} ({len(group)} images):")
            for path in group:
                size_kb = path.stat().st_size / 1024
                lines.append(f"  - {path.name} ({size_kb:.1f} KB)")
            lines.append("")

    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Total images scanned:    {total_images}")
    lines.append(f"  Duplicate groups found:  {len(duplicate_groups)}")
    lines.append(f"  Total duplicate images:  {total_duplicates}")
    lines.append(f"  Unique images:           {total_images - total_duplicates}")
    lines.append("=" * 70)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def find_duplicates(
    folder_path: str,
    threshold: int = 6,
    algorithm: str = "phash",
    recursive: bool = False,
) -> list[list[Path]]:
    """Main entry point for finding duplicate images."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return []
    if not folder.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        return []
    if algorithm not in HASH_FUNCTIONS:
        print(f"Error: Unknown algorithm '{algorithm}'. Choose from: {list(HASH_FUNCTIONS.keys())}")
        return []

    print(f"Duplicate Image Finder")
    print("=" * 70)
    print(f"  Folder:     {folder}")
    print(f"  Algorithm:  {algorithm}")
    print(f"  Threshold:  {threshold}")
    print(f"  Recursive:  {recursive}")
    print("=" * 70)

    # Step 1: Collect images
    print("\n[1/3] Collecting images...")
    image_paths = collect_image_paths(folder, recursive)
    print(f"  Found {len(image_paths)} images.")

    if len(image_paths) < 2:
        print("  Not enough images to compare.")
        return []

    # Step 2: Compute hashes
    print("\n[2/3] Computing perceptual hashes...")
    hashes = compute_hashes(image_paths, algorithm)
    print(f"  Successfully hashed {len(hashes)} images.")

    # Step 3: Find duplicates
    print("\n[3/3] Finding duplicates...")
    duplicate_groups = find_duplicate_groups(hashes, threshold)

    # Save report
    report_path = save_report(
        folder, duplicate_groups, len(hashes), algorithm, threshold, recursive
    )

    # Print summary to console
    total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if not duplicate_groups:
        print("No duplicate images found.")
    else:
        for idx, group in enumerate(duplicate_groups, 1):
            print(f"\nGroup {idx} ({len(group)} images):")
            for path in group:
                size_kb = path.stat().st_size / 1024
                print(f"  - {path.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Total images scanned:    {len(hashes)}")
    print(f"  Duplicate groups found:  {len(duplicate_groups)}")
    print(f"  Total duplicate images:  {total_duplicates}")
    print(f"  Unique images:           {len(hashes) - total_duplicates}")
    print(f"\n  Report saved to: {report_path}")
    print("=" * 70)

    return duplicate_groups


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate images using perceptual hashing."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder to scan for duplicate images.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=6,
        help="Hamming distance threshold (0=exact match, higher=more lenient). Default: 6",
    )
    parser.add_argument(
        "--algorithm", "-a",
        choices=list(HASH_FUNCTIONS.keys()),
        default="phash",
        help="Hashing algorithm to use. Default: phash",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Scan subfolders recursively.",
    )

    args = parser.parse_args()

    find_duplicates(
        folder_path=args.folder,
        threshold=args.threshold,
        algorithm=args.algorithm,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
