import argparse
import pathlib
import random
import shutil

_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class DatasetSplitter:

    def __init__(
        self,
        source_folder_path: pathlib.Path,
        train_destination_path: pathlib.Path,
        val_destination_path: pathlib.Path,
        test_destination_path: pathlib.Path,
        train_percentage: int,
    ) -> None:
        if not 1 <= train_percentage <= 98:
            raise ValueError(
                f"train_percentage must be between 1 and 98, got {train_percentage}."
            )

        self._source_folder_path = source_folder_path
        self._train_destination_path = train_destination_path
        self._val_destination_path = val_destination_path
        self._test_destination_path = test_destination_path
        self._train_fraction = train_percentage / 100.0
        self._remaining_fraction = (1.0 - self._train_fraction) / 2.0

    def split(self) -> None:
        class_dirs = [entry for entry in self._source_folder_path.iterdir() if entry.is_dir()]

        if not class_dirs:
            print(f"No class subfolders found in {self._source_folder_path}. Nothing to split.")
            return

        for class_dir in sorted(class_dirs):
            name = class_dir.name
            images = [
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS
            ]

            if not images:
                print(f"  [{name}] No images found. Skipping.")
                continue

            random.shuffle(images)

            n_total = len(images)
            train_end = int(n_total * self._train_fraction)
            val_end = train_end + int(n_total * self._remaining_fraction)

            train_imgs = images[:train_end]
            val_imgs = images[train_end:val_end]
            test_imgs = images[val_end:]

            self._copy_to(train_imgs, self._train_destination_path / name)
            self._copy_to(val_imgs, self._val_destination_path / name)
            self._copy_to(test_imgs, self._test_destination_path / name)

            print(f"  [{name}]  train: {len(train_imgs)}  val: {len(val_imgs)}  test: {len(test_imgs)}")

    def _copy_to(self, paths: list, dest: pathlib.Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, dest / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train-percent", type=int, required=True)

    args = parser.parse_args()

    splitter = DatasetSplitter(
        source_folder_path=pathlib.Path(args.source),
        train_destination_path=pathlib.Path(args.train),
        val_destination_path=pathlib.Path(args.val),
        test_destination_path=pathlib.Path(args.test),
        train_percentage=args.train_percent,
    )

    print(f"Splitting dataset from: {args.source}")
    print(f"  Train ({args.train_percent}%) -> {args.train}")
    print(f"  Val   ({(100 - args.train_percent) // 2}%) -> {args.val}")
    print(f"  Test  ({(100 - args.train_percent) // 2}%) -> {args.test}")
    print()

    splitter.split()

    print("\nDone.")
