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
    )

    # --- 4. Train ---
    print("=" * 70)
    print("Training")
    print("=" * 70)

    my_model.train_on_dataset(
        dataset_root_path / "train"
    )

    checkpoint_file_path = pathlib.Path("checkpoint.pth")
    my_model.save_checkpoint(checkpoint_file_path)
    print(f"\n  Checkpoint saved to {checkpoint_file_path}")

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
    print("=" * 70)


if __name__ == "__main__":
    main()
