import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


EPOCH_PATTERN = re.compile(
    r"Epoch \[(\d+)/\d+\]\s+loss:\s*([\d.]+)\s+accuracy:\s*([\d.]+)%\s+val_loss:\s*([\d.]+)"
    r"(?:\s+val_accuracy:\s*([\d.]+)%)?"
)
FOLD_HEADER_PATTERN = re.compile(r"Fold \[(\d+) / \d+\]")
FOLD_RESULT_PATTERN = re.compile(
    r"Fold (\d+) result:\s+val_loss:\s*([\d.]+)\s+val_accuracy:\s*([\d.]+)%"
)


def locate_log_file(results_path: Path) -> Path:
    """Return the single .out log file inside results_path, or raise if ambiguous."""
    candidate_files = sorted(results_path.glob("*.out"))
    if not candidate_files:
        raise FileNotFoundError(f"No .out log file found in {results_path}")
    if len(candidate_files) > 1:
        # Prefer the most recently modified one
        candidate_files = sorted(candidate_files, key=lambda file_path: file_path.stat().st_mtime)
    return candidate_files[-1]


def parse_log_file(log_file_path: Path) -> dict:
    """
    Parse a Slurm training log and return structured epoch data.

    Returns a dict with:
        mode: "cross_validation" | "single"
        folds: list of dicts, each with keys:
            fold_number, epochs, train_losses, train_accuracies, val_losses
            final_val_loss, final_val_accuracy (cross-val only)
    """
    log_text = log_file_path.read_text(encoding="utf-8", errors="replace")

    is_cross_validation = bool(FOLD_HEADER_PATTERN.search(log_text))

    fold_data_list = []

    if is_cross_validation:
        # Split log into per-fold sections
        fold_section_boundaries = [
            match.start() for match in FOLD_HEADER_PATTERN.finditer(log_text)
        ]
        fold_section_boundaries.append(len(log_text))

        for section_index, section_start in enumerate(fold_section_boundaries[:-1]):
            section_end = fold_section_boundaries[section_index + 1]
            fold_section_text = log_text[section_start:section_end]

            fold_header_match = FOLD_HEADER_PATTERN.search(fold_section_text)
            fold_number = int(fold_header_match.group(1))

            epoch_data = _extract_epoch_rows(fold_section_text)
            fold_result_match = FOLD_RESULT_PATTERN.search(fold_section_text)

            final_val_loss = float(fold_result_match.group(2)) if fold_result_match else None
            final_val_accuracy = float(fold_result_match.group(3)) if fold_result_match else None

            fold_data_list.append({
                "fold_number": fold_number,
                **epoch_data,
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_accuracy,
            })
    else:
        epoch_data = _extract_epoch_rows(log_text)
        fold_data_list.append({
            "fold_number": None,
            **epoch_data,
            "final_val_loss": None,
            "final_val_accuracy": None,
        })

    return {
        "mode": "cross_validation" if is_cross_validation else "single",
        "folds": fold_data_list,
    }


def _extract_epoch_rows(text: str) -> dict:
    """Extract per-epoch metrics from a block of log text."""
    epoch_numbers = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch_match in EPOCH_PATTERN.finditer(text):
        epoch_numbers.append(int(epoch_match.group(1)))
        train_losses.append(float(epoch_match.group(2)))
        train_accuracies.append(float(epoch_match.group(3)))
        val_losses.append(float(epoch_match.group(4)))
        raw_val_accuracy = epoch_match.group(5)
        val_accuracies.append(float(raw_val_accuracy) if raw_val_accuracy is not None else None)

    return {
        "epochs": epoch_numbers,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }


def load_results_json(results_directory: Path) -> Optional[dict]:
    """Load results.json from the directory if it exists."""
    results_json_path = results_directory / "results.json"
    if results_json_path.exists():
        with open(results_json_path, encoding="utf-8") as json_file:
            return json.load(json_file)
    return None


def plot_training_curves(parsed_data: dict, results_json: Optional[dict], output_path: Path, version_label: str) -> None:
    """Render and save training curve plots plus a per-class F1 bar chart."""
    training_mode = parsed_data["mode"]
    fold_list = parsed_data["folds"]

    per_class_results = None
    if results_json:
        per_class_results = results_json.get("results", {}).get("per_class_results")

    has_per_class = bool(per_class_results)
    figure_height = 10 if has_per_class else 9
    figure = plt.figure(figsize=(16, figure_height))
    figure.suptitle(f"Training Results — {version_label}", fontsize=14, fontweight="bold", y=0.98)

    if has_per_class:
        grid_spec = gridspec.GridSpec(3, 2, figure=figure, hspace=0.55, wspace=0.30,
                                      height_ratios=[1, 1, 1.1])
    else:
        grid_spec = gridspec.GridSpec(2, 2, figure=figure, hspace=0.45, wspace=0.30)

    train_loss_axis = figure.add_subplot(grid_spec[0, 0])
    val_loss_axis   = figure.add_subplot(grid_spec[0, 1])
    train_acc_axis  = figure.add_subplot(grid_spec[1, 0])
    val_acc_axis    = figure.add_subplot(grid_spec[1, 1])

    fold_color_palette = plt.colormaps["tab10"]

    all_train_losses_padded    = []
    all_val_losses_padded      = []
    all_train_accuracies_padded = []
    all_val_accuracies_padded  = []
    max_epoch_count = max(len(fold["epochs"]) for fold in fold_list)

    has_val_accuracy_data = any(
        any(value is not None for value in fold["val_accuracies"])
        for fold in fold_list
    )

    for fold_index, fold_data in enumerate(fold_list):
        epoch_numbers    = fold_data["epochs"]
        train_losses     = fold_data["train_losses"]
        val_losses       = fold_data["val_losses"]
        train_accuracies = fold_data["train_accuracies"]
        val_accuracies   = fold_data["val_accuracies"]

        if not epoch_numbers:
            continue

        if training_mode == "cross_validation":
            fold_number = fold_data["fold_number"]
            fold_color  = fold_color_palette(fold_index / max(len(fold_list), 1))
            fold_label  = f"Fold {fold_number}"
            line_alpha  = 0.6
            line_width  = 1.4
        else:
            fold_color = "steelblue"
            fold_label = "Run"
            line_alpha = 1.0
            line_width = 2.0

        train_loss_axis.plot(epoch_numbers, train_losses, color=fold_color, alpha=line_alpha,
                             linewidth=line_width, label=fold_label)
        val_loss_axis.plot(epoch_numbers, val_losses, color=fold_color, alpha=line_alpha,
                           linewidth=line_width, label=fold_label)
        train_acc_axis.plot(epoch_numbers, train_accuracies, color=fold_color, alpha=line_alpha,
                            linewidth=line_width, label=fold_label)

        if has_val_accuracy_data:
            clean_val_accuracies = [value if value is not None else np.nan for value in val_accuracies]
            val_acc_axis.plot(epoch_numbers, clean_val_accuracies, color=fold_color, alpha=line_alpha,
                              linewidth=line_width, label=fold_label)

        # Collect for mean computation
        padded_train_losses      = train_losses      + [np.nan] * (max_epoch_count - len(train_losses))
        padded_val_losses        = val_losses        + [np.nan] * (max_epoch_count - len(val_losses))
        padded_train_accuracies  = train_accuracies  + [np.nan] * (max_epoch_count - len(train_accuracies))
        padded_val_accuracies    = (
            [value if value is not None else np.nan for value in val_accuracies]
            + [np.nan] * (max_epoch_count - len(val_accuracies))
        )
        all_train_losses_padded.append(padded_train_losses)
        all_val_losses_padded.append(padded_val_losses)
        all_train_accuracies_padded.append(padded_train_accuracies)
        all_val_accuracies_padded.append(padded_val_accuracies)

    # Draw mean curve for cross-validation
    if training_mode == "cross_validation" and len(fold_list) > 1:
        mean_epoch_range         = list(range(1, max_epoch_count + 1))
        mean_train_losses        = np.nanmean(all_train_losses_padded, axis=0)
        mean_val_losses          = np.nanmean(all_val_losses_padded, axis=0)
        mean_train_accuracies    = np.nanmean(all_train_accuracies_padded, axis=0)
        mean_val_accuracies      = np.nanmean(all_val_accuracies_padded, axis=0)

        train_loss_axis.plot(mean_epoch_range, mean_train_losses, color="black",
                             linewidth=2.2, linestyle="--", label="Mean", zorder=5)
        val_loss_axis.plot(mean_epoch_range, mean_val_losses, color="black",
                           linewidth=2.2, linestyle="--", label="Mean", zorder=5)
        train_acc_axis.plot(mean_epoch_range, mean_train_accuracies, color="black",
                            linewidth=2.2, linestyle="--", label="Mean", zorder=5)
        if has_val_accuracy_data:
            val_acc_axis.plot(mean_epoch_range, mean_val_accuracies, color="black",
                              linewidth=2.2, linestyle="--", label="Mean", zorder=5)

    if not has_val_accuracy_data:
        val_acc_axis.text(
            0.5, 0.5, "Val accuracy not available\n(old log format)",
            ha="center", va="center", transform=val_acc_axis.transAxes,
            fontsize=9, color="gray", style="italic",
        )

    # Style the four curve axes
    _style_axis(train_loss_axis, title="Train Loss",        xlabel="Epoch", ylabel="Loss")
    _style_axis(val_loss_axis,   title="Validation Loss",   xlabel="Epoch", ylabel="Loss")
    _style_axis(train_acc_axis,  title="Train Accuracy",    xlabel="Epoch", ylabel="Accuracy (%)")
    _style_axis(val_acc_axis,    title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")

    if training_mode == "cross_validation":
        for axis in [train_loss_axis, val_loss_axis, train_acc_axis, val_acc_axis]:
            axis.legend(fontsize=7, loc="upper right")

    # Per-class F1 bar chart (bottom row, spanning both columns)
    if has_per_class:
        per_class_axis = figure.add_subplot(grid_spec[2, :])
        class_names_sorted = sorted(per_class_results.keys())
        f1_scores      = [per_class_results[name]["f1"]        for name in class_names_sorted]
        precision_vals = [per_class_results[name]["precision"] for name in class_names_sorted]
        recall_vals    = [per_class_results[name]["recall"]    for name in class_names_sorted]

        bar_positions = np.arange(len(class_names_sorted))
        bar_width = 0.26

        per_class_axis.bar(bar_positions - bar_width, precision_vals, width=bar_width,
                           label="Precision", color="steelblue",   alpha=0.85)
        per_class_axis.bar(bar_positions,             f1_scores,     width=bar_width,
                           label="F1",        color="seagreen",    alpha=0.85)
        per_class_axis.bar(bar_positions + bar_width, recall_vals,   width=bar_width,
                           label="Recall",    color="darkorange",  alpha=0.85)

        per_class_axis.set_xticks(bar_positions)
        per_class_axis.set_xticklabels(class_names_sorted, rotation=20, ha="right", fontsize=8)
        per_class_axis.set_ylim(0, 1.05)

        test_accuracy = results_json.get("results", {}).get("test_accuracy")
        title_suffix = f"  (overall test accuracy: {test_accuracy * 100:.2f}%)" if test_accuracy is not None else ""
        _style_axis(per_class_axis, title=f"Per-Class Metrics — Test Set{title_suffix}",
                    xlabel="Class", ylabel="Score")
        per_class_axis.legend(fontsize=8, loc="lower right")

    # Summary annotation
    if results_json:
        cross_validation_section = results_json.get("cross_validation")
        annotation_lines = []
        if cross_validation_section:
            mean_val_loss     = cross_validation_section.get("mean_val_loss")
            std_val_loss      = cross_validation_section.get("std_val_loss")
            mean_val_accuracy = cross_validation_section.get("mean_val_accuracy")
            std_val_accuracy  = cross_validation_section.get("std_val_accuracy")
            if mean_val_loss is not None:
                annotation_lines.append(f"Mean Val Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
            if mean_val_accuracy is not None:
                annotation_lines.append(f"Mean Val Acc: {mean_val_accuracy * 100:.2f}% ± {std_val_accuracy * 100:.2f}%")
        if not has_per_class:
            test_accuracy = results_json.get("results", {}).get("test_accuracy")
            if test_accuracy is not None:
                annotation_lines.append(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        if annotation_lines:
            figure.text(
                0.5, -0.02, "  |  ".join(annotation_lines),
                ha="center", va="top", fontsize=9,
                style="italic", color="dimgray",
                transform=figure.transFigure,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot saved to: {output_path}")


def _style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=11, fontweight="bold")
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.tick_params(labelsize=8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def resolve_target(target_argument: Optional[str]) -> tuple:
    """
    Resolve the user-supplied target to (results_directory, log_file_path).

    Accepts:
      - Path to a results.json file
      - Path to a .out log file directly
      - Path to a results version directory (e.g. results/v9)
      - None → uses the most recently modified version directory
    """
    base_results_dir = Path(__file__).parent / "results"

    if target_argument is None:
        version_directories = [
            candidate for candidate in base_results_dir.iterdir()
            if candidate.is_dir() and list(candidate.glob("*.out"))
        ]
        if not version_directories:
            raise FileNotFoundError(f"No result directories with .out files found in {base_results_dir}")
        results_directory = max(version_directories, key=lambda candidate: candidate.stat().st_mtime)
        log_file_path = locate_log_file(results_directory)
        return results_directory, log_file_path

    target_path = Path(target_argument)
    if not target_path.is_absolute():
        # Try relative to script location first, then CWD
        relative_to_script = Path(__file__).parent / target_path
        if relative_to_script.exists():
            target_path = relative_to_script

    if target_path.is_file() and target_path.name == "results.json":
        results_directory = target_path.parent
        log_file_path = locate_log_file(results_directory)
        return results_directory, log_file_path

    if target_path.is_file() and target_path.suffix == ".out":
        return target_path.parent, target_path

    if target_path.is_dir():
        log_file_path = locate_log_file(target_path)
        return target_path, log_file_path

    raise ValueError(
        f"Cannot resolve target: {target_argument!r}. "
        "Provide a results.json file, a .out log file, or a results version directory."
    )


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Plot training curves (loss, val loss, train accuracy) from a Slurm training log."
    )
    argument_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "Path to a results.json file, a .out log file, a results version directory "
            "(e.g. results/v9), or omit to auto-select the most recent version."
        ),
    )
    argument_parser.add_argument(
        "--output",
        default=None,
        help="Output image file path. Defaults to <results_dir>/training_curves.png",
    )
    parsed_args = argument_parser.parse_args()

    try:
        results_directory, log_file_path = resolve_target(parsed_args.target)
    except (FileNotFoundError, ValueError) as resolution_error:
        print(f"Error: {resolution_error}", file=sys.stderr)
        sys.exit(1)

    version_label = results_directory.name
    print(f"Parsing log: {log_file_path}")

    parsed_data = parse_log_file(log_file_path)
    results_json = load_results_json(results_directory)

    total_epoch_count = sum(len(fold["epochs"]) for fold in parsed_data["folds"])
    fold_count = len(parsed_data["folds"])
    print(
        f"Mode: {parsed_data['mode']}  |  "
        f"Folds: {fold_count}  |  "
        f"Total epochs parsed: {total_epoch_count}"
    )

    output_image_path = Path(parsed_args.output) if parsed_args.output else results_directory / "training_curves.png"
    plot_training_curves(parsed_data, results_json, output_image_path, version_label)


if __name__ == "__main__":
    main()
