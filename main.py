import datetime
import json
import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim

from model.cnn import CNN
from model.cross_validator import CrossValidator
from model.focal_loss import FocalLoss
from model.pretrained_model import PretrainedModel
from model.preprocessing_pipeline import PreprocessingPipeline


_OPTIMIZER_CLASSES = {
    "Adam":    torch.optim.Adam,
    "AdamW":   torch.optim.AdamW,
    "SGD":     torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}


def _build_loss_function(
    loss_function_name: str,
    class_names: list,
    training_dataset_path: pathlib.Path,
    use_class_weights: bool,
    label_smoothing: float = 0.0,
    use_focal_loss: bool = False,
    focal_loss_gamma: float = 2.0,
) -> nn.Module:
    """
    Builds the loss function, optionally with inverse-frequency class weights,
    label smoothing, and focal loss.

    When use_class_weights is True, counts images per class in the training
    folder and computes weights as (1 / class_count), normalized so they sum
    to the number of classes. This causes the loss to penalize errors on
    minority classes proportionally more than majority ones.

    When use_focal_loss is True, uses FocalLoss instead of CrossEntropyLoss.
    Focal loss down-weights easy examples and focuses training on hard ones,
    which helps when certain classes are consistently ignored by the model.

    :param loss_function_name: Either 'CrossEntropyLoss' or 'NLLLoss'.
    :param class_names: Ordered list of class label strings matching config order.
    :param training_dataset_path: Path to the training split folder.
    :param use_class_weights: Whether to apply inverse-frequency class weights.
    :param label_smoothing: Label smoothing factor in [0.0, 1.0). Defaults to 0.0.
    :param use_focal_loss: Whether to use FocalLoss instead of CrossEntropyLoss.
    :param focal_loss_gamma: Focusing exponent for focal loss. Defaults to 2.0.
    :return: An instantiated loss function module.
    """
    if loss_function_name == "NLLLoss":
        return nn.NLLLoss()

    class_weight_tensor = None

    if use_class_weights:
        class_sample_counts = []
        for class_name in class_names:
            class_folder = training_dataset_path / class_name
            image_count = sum(
                1 for file_path in class_folder.rglob("*")
                if file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
            )
            class_sample_counts.append(image_count)

        inverse_frequencies = [1.0 / count for count in class_sample_counts]
        total_inverse = sum(inverse_frequencies)
        num_classes = len(class_names)
        normalized_weights = [value * num_classes / total_inverse for value in inverse_frequencies]
        class_weight_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

        print("  Class weights (inverse-frequency):")
        for class_name, weight, count in zip(class_names, normalized_weights, class_sample_counts):
            print(f"    {class_name:<30}  {count:>4} samples  weight: {weight:.4f}")

    if use_focal_loss:
        print(f"  Loss function: FocalLoss  (gamma={focal_loss_gamma})")
        return FocalLoss(
            gamma=focal_loss_gamma,
            class_weights=class_weight_tensor,
            label_smoothing=label_smoothing,
        )

    if class_weight_tensor is not None:
        return nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=label_smoothing)

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _build_versioned_experiment_folder(results_root_path: pathlib.Path) -> pathlib.Path:
    """
    Creates and returns the next versioned experiment folder (e.g. results/v3/).

    :param results_root_path: Root directory where all versioned folders live.
    :return: Path to the newly created experiment folder.
    """
    results_root_path.mkdir(exist_ok=True)
    existing_version_numbers = [
        int(folder.name[1:])
        for folder in results_root_path.iterdir()
        if folder.is_dir() and folder.name.startswith("v") and folder.name[1:].isdigit()
    ]
    next_version_number = max(existing_version_numbers, default=0) + 1
    experiment_folder_path = results_root_path / f"v{next_version_number}"
    experiment_folder_path.mkdir()
    return experiment_folder_path


def _build_cnn_factory(
    model_config: dict,
    training_config: dict,
    augmentations_config: dict,
    optimizer_class: type,
    loss_function: nn.Module,
) -> callable:
    """
    Returns a zero-argument factory that constructs a fresh CNN each time it is called.

    Used by CrossValidator so each fold starts with independently initialised weights.

    :return: Callable with no arguments that returns a new CNN instance.
    """
    def factory():
        return CNN(
            in_size=tuple(model_config["in_size"]),
            class_names=model_config["class_names"],
            channels=model_config["channels"],
            pool_every=model_config["pool_every"],
            hidden_dims=model_config["hidden_dims"],
            num_epochs=training_config["num_epochs"],
            optimizer_class=optimizer_class,
            loss_function=loss_function,
            conv_kernel_size=model_config["conv_kernel_size"],
            pooling_type=model_config["pooling_type"],
            pool_kernel_size=model_config["pool_kernel_size"],
            image_normalization_mean=model_config["image_normalization_mean"],
            image_normalization_std=model_config["image_normalization_std"],
            num_dataloader_workers=training_config["num_dataloader_workers"],
            batch_size=training_config["batch_size"],
            activation=model_config["activation"],
            use_batchnorm=model_config["use_batchnorm"],
            dropout_probability=model_config["dropout_probability"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            early_stopping_patience=training_config["early_stopping_patience"],
            augmentations_config=augmentations_config,
        )
    return factory


def _build_pretrained_factory(
    model_config: dict,
    training_config: dict,
    pretrained_config: dict,
    augmentations_config: dict,
    optimizer_class: type,
    loss_function: nn.Module,
) -> callable:
    """
    Returns a zero-argument factory that constructs a fresh PretrainedModel each time.

    Used by CrossValidator so each fold starts with independently initialised weights.
    The pretrained backbone is re-downloaded (or loaded from cache) on each call.

    :return: Callable with no arguments that returns a new PretrainedModel instance.
    """
    def factory():
        return PretrainedModel(
            backbone_name=pretrained_config["backbone"],
            class_names=model_config["class_names"],
            hidden_dims=pretrained_config.get("hidden_dims", []),
            dropout_probability=pretrained_config.get("dropout_probability", 0.5),
            freeze_backbone=pretrained_config.get("freeze_backbone", True),
            unfreeze_after_epoch=pretrained_config.get("unfreeze_after_epoch", 0),
            learning_rate=training_config["learning_rate"],
            backbone_learning_rate=pretrained_config.get("backbone_learning_rate", 1e-5),
            weight_decay=training_config["weight_decay"],
            num_epochs=training_config["num_epochs"],
            batch_size=training_config["batch_size"],
            num_dataloader_workers=training_config["num_dataloader_workers"],
            optimizer_class=optimizer_class,
            loss_function=loss_function,
            image_normalization_mean=model_config["image_normalization_mean"],
            image_normalization_std=model_config["image_normalization_std"],
            in_size=tuple(model_config["in_size"]),
            early_stopping_patience=training_config["early_stopping_patience"],
            augmentations_config=augmentations_config,
        )
    return factory


def _run_cross_validation(
    model_config: dict,
    training_config: dict,
    augmentations_config: dict,
    cross_validation_config: dict,
    dataset_root_path: pathlib.Path,
    model_factory: callable,
    model_label: str,
    raw_config: dict,
) -> None:
    """
    Runs K-fold cross-validation and saves the results to a versioned folder.

    :param model_config: Model hyperparameter block from config.json.
    :param training_config: Training hyperparameter block from config.json.
    :param augmentations_config: Augmentations block from config.json.
    :param cross_validation_config: Cross-validation block from config.json.
    :param dataset_root_path: Root path containing train/, val/, and test/ splits.
    :param model_factory: Zero-argument callable that returns a fresh model instance.
    :param model_label: Human-readable label for the model type, included in results.json.
    :param raw_config: The full config dict as loaded from config.json, saved as a snapshot.
    """
    experiment_folder_path = _build_versioned_experiment_folder(pathlib.Path("results"))
    config_snapshot_path = experiment_folder_path / "config_snapshot.json"
    with config_snapshot_path.open("w", encoding="utf-8") as config_snapshot_file:
        json.dump(raw_config, config_snapshot_file, indent=4)
    print(f"  Config snapshot saved to {config_snapshot_path}")

    print("=" * 70)
    print(f"Cross-Validation  ({cross_validation_config['num_folds']} folds)")
    print("=" * 70)

    cross_validator = CrossValidator(
        num_folds=cross_validation_config["num_folds"],
        in_size=tuple(model_config["in_size"]),
        image_normalization_mean=model_config["image_normalization_mean"],
        image_normalization_std=model_config["image_normalization_std"],
        batch_size=training_config["batch_size"],
        num_dataloader_workers=training_config["num_dataloader_workers"],
        model_factory=model_factory,
    )

    training_start_time = datetime.datetime.now()
    cv_output = cross_validator.run(
        train_dataset_path=dataset_root_path / "train",
        val_dataset_path=dataset_root_path / "val",
        test_dataset_path=dataset_root_path / "test",
    )
    training_duration_seconds = (datetime.datetime.now() - training_start_time).total_seconds()

    final_model = cv_output["final_model"]
    checkpoint_file_path = experiment_folder_path / "checkpoint.pth"
    final_model.save_checkpoint(checkpoint_file_path)
    print(f"\n  Checkpoint saved to {checkpoint_file_path}")

    cv_metrics = cv_output["cross_validation"]
    test_results = cv_output["test_results"]

    experiment_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "training_duration_seconds": training_duration_seconds,
        "mode": "cross_validation",
        "model_type": model_label,
        "hyperparameters": {
            "model": model_config,
            "training": training_config,
            "cross_validation": cross_validation_config,
        },
        "augmentations": final_model.augmentation_description(),
        "use_class_weights": training_config.get("use_class_weights", False),
        "use_focal_loss": training_config.get("use_focal_loss", False),
        "focal_loss_gamma": training_config.get("focal_loss_gamma", 2.0) if training_config.get("use_focal_loss", False) else None,
        "cross_validation": {
            "num_folds": cv_metrics["num_folds"],
            "per_fold_results": cv_metrics["per_fold_results"],
            "mean_val_loss": cv_metrics["mean_val_loss"],
            "std_val_loss": cv_metrics["std_val_loss"],
            "mean_val_accuracy": cv_metrics["mean_val_accuracy"],
            "std_val_accuracy": cv_metrics["std_val_accuracy"],
        },
        "results": {
            "test_accuracy": test_results["test_accuracy"],
            "per_class_results": test_results["per_class_results"],
            "raw_predictions": test_results["raw_predictions"],
        },
    }

    experiment_file_path = experiment_folder_path / "results.json"
    with experiment_file_path.open("w", encoding="utf-8") as experiment_file:
        json.dump(experiment_record, experiment_file, indent=4)

    print(f"  Results saved to {experiment_file_path}")


def main() -> None:
    config_file_path = pathlib.Path("config.json")
    dataset_root_path = pathlib.Path("dataset")

    # --- 1. Load config ---
    with config_file_path.open(encoding="utf-8") as config_file:
        loaded_config = json.load(config_file)

    model_config = loaded_config["model"]
    training_config = loaded_config["training"]
    augmentations_config = loaded_config.get("augmentations", {"enabled": False})
    cross_validation_config = loaded_config.get("cross_validation", {"enabled": False, "num_folds": 5})
    pretrained_config = loaded_config.get("pretrained_model", {"enabled": False})

    use_pretrained = pretrained_config.get("enabled", False)

    # --- 2. Preprocess ---
    preprocessing_pipeline = PreprocessingPipeline(
        dataset_root_path=dataset_root_path,
        config_file_path=config_file_path,
    )
    dataset_validation_report = preprocessing_pipeline.run_full_pipeline()

    if not dataset_validation_report.is_valid:
        raise RuntimeError("Dataset validation failed. Aborting training.")

    optimizer_class = _OPTIMIZER_CLASSES[training_config["optimizer"]]
    loss_function = _build_loss_function(
        loss_function_name=training_config["loss_function"],
        class_names=model_config["class_names"],
        training_dataset_path=dataset_root_path / "train",
        use_class_weights=training_config.get("use_class_weights", False),
        label_smoothing=training_config.get("label_smoothing", 0.0),
        use_focal_loss=training_config.get("use_focal_loss", False),
        focal_loss_gamma=training_config.get("focal_loss_gamma", 2.0),
    )

    # --- 3. Build model factory ---
    if use_pretrained:
        backbone_name = pretrained_config["backbone"]
        print(f"\nUsing pretrained backbone: {backbone_name}")
        model_factory = _build_pretrained_factory(
            model_config=model_config,
            training_config=training_config,
            pretrained_config=pretrained_config,
            augmentations_config=augmentations_config,
            optimizer_class=optimizer_class,
            loss_function=loss_function,
        )
        model_label = f"pretrained:{backbone_name}"
    else:
        model_factory = _build_cnn_factory(
            model_config=model_config,
            training_config=training_config,
            augmentations_config=augmentations_config,
            optimizer_class=optimizer_class,
            loss_function=loss_function,
        )
        model_label = "cnn"

    # --- 4. Branch: cross-validation or standard training ---
    if cross_validation_config["enabled"]:
        _run_cross_validation(
            model_config=model_config,
            training_config=training_config,
            augmentations_config=augmentations_config,
            cross_validation_config=cross_validation_config,
            dataset_root_path=dataset_root_path,
            model_factory=model_factory,
            model_label=model_label,
            raw_config=loaded_config,
        )
        return

    # --- Standard training path ---

    # --- 5. Create versioned experiment folder and save config snapshot before training ---
    experiment_folder_path = _build_versioned_experiment_folder(pathlib.Path("results"))
    config_snapshot_path = experiment_folder_path / "config_snapshot.json"
    with config_snapshot_path.open("w", encoding="utf-8") as config_snapshot_file:
        json.dump(loaded_config, config_snapshot_file, indent=4)
    print(f"  Config snapshot saved to {config_snapshot_path}")

    # --- 6. Build model ---
    my_model = model_factory()

    # --- 7. Train ---
    print("=" * 70)
    print("Training")
    print("=" * 70)

    training_start_time = datetime.datetime.now()
    my_model.train_on_dataset(
        dataset_root_path / "train",
        val_dataset_path=dataset_root_path / "val",
    )
    training_duration_seconds = (datetime.datetime.now() - training_start_time).total_seconds()

    # --- 8. Validate ---
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)

    validation_results = my_model.validate_on_dataset(
        dataset_root_path / "val"
    )

    print(f"  Validation loss:     {validation_results['val_loss']:.4f}")
    print(f"  Validation accuracy: {validation_results['val_accuracy'] * 100:.2f}%")

    # --- 9. Test ---
    print("\n" + "=" * 70)
    print("Testing")
    print("=" * 70)

    test_results = my_model.test_on_dataset(
        dataset_root_path / "test"
    )

    print(f"  Test accuracy: {test_results['test_accuracy'] * 100:.2f}%")
    print()
    print(f"  {'Class':<25} {'Samples':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 63)
    for class_name, class_metrics in test_results["per_class_results"].items():
        print(
            f"  {class_name:<25}"
            f"  {class_metrics['total_samples_tested']:>6}"
            f"  {class_metrics['precision']:>9.4f}"
            f"  {class_metrics['recall']:>7.4f}"
            f"  {class_metrics['f1']:>7.4f}"
        )
    print("=" * 70)

    # --- 10. Save checkpoint ---
    checkpoint_file_path = experiment_folder_path / "checkpoint.pth"
    my_model.save_checkpoint(checkpoint_file_path)
    print(f"\n  Checkpoint saved to {checkpoint_file_path}")

    # --- 11. Save experiment results ---
    experiment_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "training_duration_seconds": training_duration_seconds,
        "mode": "standard",
        "model_type": model_label,
        "hyperparameters": {
            "model": model_config,
            "training": training_config,
        },
        "augmentations": my_model.augmentation_description(),
        "use_class_weights": training_config.get("use_class_weights", False),
        "use_focal_loss": training_config.get("use_focal_loss", False),
        "focal_loss_gamma": training_config.get("focal_loss_gamma", 2.0) if training_config.get("use_focal_loss", False) else None,
        "results": {
            "val_loss": validation_results["val_loss"],
            "val_accuracy": validation_results["val_accuracy"],
            "test_accuracy": test_results["test_accuracy"],
            "per_class_results": test_results["per_class_results"],
            "raw_predictions": test_results["raw_predictions"],
        },
    }

    experiment_file_path = experiment_folder_path / "results.json"
    with experiment_file_path.open("w", encoding="utf-8") as experiment_file:
        json.dump(experiment_record, experiment_file, indent=4)

    print(f"  Results saved to {experiment_file_path}")


if __name__ == "__main__":
    main()
