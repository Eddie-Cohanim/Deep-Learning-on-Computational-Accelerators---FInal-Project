import datetime
import json
import pathlib

import torch.nn as nn
import torch.optim

from model.cnn import CNN
from model.preprocessing_pipeline import PreprocessingPipeline


_OPTIMIZER_CLASSES = {
    "Adam":    torch.optim.Adam,
    "AdamW":   torch.optim.AdamW,
    "SGD":     torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
}

_LOSS_FUNCTION_INSTANCES = {
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "NLLLoss":          nn.NLLLoss(),
}


def main() -> None:
    config_file_path = pathlib.Path("config.json")
    dataset_root_path = pathlib.Path("dataset")

    # --- 1. Load config ---
    with config_file_path.open(encoding="utf-8") as config_file:
        loaded_config = json.load(config_file)

    model_config = loaded_config["model"]
    training_config = loaded_config["training"]

    # --- 2. Preprocess ---
    preprocessing_pipeline = PreprocessingPipeline(
        dataset_root_path=dataset_root_path,
        config_file_path=config_file_path,
    )
    dataset_validation_report = preprocessing_pipeline.run_full_pipeline()

    if not dataset_validation_report.is_valid:
        raise RuntimeError("Dataset validation failed. Aborting training.")

    # --- 3. Build CNN ---
    optimizer_class = _OPTIMIZER_CLASSES[training_config["optimizer"]]
    loss_function = _LOSS_FUNCTION_INSTANCES[training_config["loss_function"]]

    my_model = CNN(
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
    )

    # --- 4. Train ---
    print("=" * 70)
    print("Training")
    print("=" * 70)

    my_model.train_on_dataset(
        dataset_root_path / "train",
        val_dataset_path=dataset_root_path / "val",
    )

    # --- 5. Validate ---
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)

    validation_results = my_model.validate_on_dataset(
        dataset_root_path / "val"
    )

    print(f"  Validation loss:     {validation_results['val_loss']:.4f}")
    print(f"  Validation accuracy: {validation_results['val_accuracy'] * 100:.2f}%")

    # --- 6. Test ---
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

    # --- 7. Create versioned experiment folder ---
    results_root_path = pathlib.Path("results")
    results_root_path.mkdir(exist_ok=True)

    existing_version_numbers = [
        int(folder.name[1:])
        for folder in results_root_path.iterdir()
        if folder.is_dir() and folder.name.startswith("v") and folder.name[1:].isdigit()
    ]
    next_version_number = max(existing_version_numbers, default=0) + 1
    experiment_folder_path = results_root_path / f"v{next_version_number}"
    experiment_folder_path.mkdir()

    # --- 8. Save checkpoint ---
    checkpoint_file_path = experiment_folder_path / "checkpoint.pth"
    my_model.save_checkpoint(checkpoint_file_path)
    print(f"\n  Checkpoint saved to {checkpoint_file_path}")

    # --- 9. Save experiment results ---
    experiment_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hyperparameters": {
            "model": model_config,
            "training": training_config,
        },
        "results": {
            "val_loss": validation_results["val_loss"],
            "val_accuracy": validation_results["val_accuracy"],
            "test_accuracy": test_results["test_accuracy"],
            "per_class_results": test_results["per_class_results"],
        },
    }

    experiment_file_path = experiment_folder_path / "results.json"
    with experiment_file_path.open("w", encoding="utf-8") as experiment_file:
        json.dump(experiment_record, experiment_file, indent=4)

    print(f"  Results saved to {experiment_file_path}")


if __name__ == "__main__":
    main()
