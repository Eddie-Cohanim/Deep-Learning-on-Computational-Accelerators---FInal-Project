import os
import pathlib


CROPPED_IMAGES_ROOT = pathlib.Path("dataset") / "cropped images"


def rename_images_in_label_directory(label_directory: pathlib.Path) -> None:
    label_name = label_directory.name
    images_subdirectory = label_directory / "cropped"

    if not images_subdirectory.is_dir():
        print(f"Skipping '{label_name}': no 'cropped' subdirectory found.")
        return

    image_files = sorted(
        entry for entry in images_subdirectory.iterdir() if entry.is_file()
    )

    for index, image_file in enumerate(image_files, start=1):
        new_name = f"{label_name}{index}{image_file.suffix}"
        new_path = images_subdirectory / new_name

        if image_file.name == new_name:
            continue

        if new_path.exists():
            print(f"  WARNING: '{new_path}' already exists, skipping '{image_file.name}'.")
            continue

        image_file.rename(new_path)

    print(f"Renamed {len(image_files)} images in '{label_name}'.")


def main() -> None:
    if not CROPPED_IMAGES_ROOT.is_dir():
        print(f"Error: '{CROPPED_IMAGES_ROOT}' does not exist.")
        return

    label_directories = sorted(
        entry for entry in CROPPED_IMAGES_ROOT.iterdir() if entry.is_dir()
    )

    if not label_directories:
        print(f"No label directories found in '{CROPPED_IMAGES_ROOT}'.")
        return

    for label_directory in label_directories:
        rename_images_in_label_directory(label_directory)


if __name__ == "__main__":
    main()
