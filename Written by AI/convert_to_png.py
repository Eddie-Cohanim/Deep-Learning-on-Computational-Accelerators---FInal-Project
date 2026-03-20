#!/usr/bin/env python3
"""Convert various image formats to PNG."""

import argparse
import sys
import io
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for non-Latin filenames
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from PIL import Image
except ImportError:
    print("Pillow not found. Install with: pip install Pillow")
    sys.exit(1)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif",
    ".webp", ".avif", ".heic", ".heif", ".ico", ".ppm", ".pgm",
    ".pbm", ".tga", ".dds", ".pcx",
}


def convert_file(src: Path, output_dir: Optional[Path], overwrite: bool) -> bool:
    dst = (output_dir or src.parent) / (src.stem + ".png")

    if dst == src:
        print(f"  skip  {src.name}  (already PNG)")
        return False

    if dst.exists() and not overwrite:
        print(f"  skip  {src.name}  (output exists, use --overwrite)")
        return False

    try:
        with Image.open(src) as img:
            img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
            img.save(dst, "PNG")
        src.unlink()
        print(f"  ok    {src.name}  ->  {dst.name}")
        return True
    except Exception as e:
        print(f"  fail  {src.name}  ({e})")
        return False


def collect_files(inputs: list[str], recursive: bool) -> list[Path]:
    files = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in p.glob(pattern):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(f)
        else:
            print(f"Warning: '{raw}' not found, skipping.")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Convert images (JPEG, AVIF, WEBP, HEIC, …) to PNG."
    )
    parser.add_argument("inputs", nargs="+", help="Files or directories to convert")
    parser.add_argument("-o", "--output-dir", help="Directory to write PNGs into")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into directories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.inputs, args.recursive)
    if not files:
        print("No supported image files found.")
        sys.exit(1)

    ok = sum(convert_file(f, output_dir, args.overwrite) for f in files)
    print(f"\nDone: {ok}/{len(files)} converted.")


if __name__ == "__main__":
    main()
