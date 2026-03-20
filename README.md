# Beer & Wine Image Classifier

A convolutional neural network for classifying types of beer and wine from images.

---

## Configuration

All model and training parameters are set in `config.json`. The main pipeline reads this file and passes the values directly to the CNN constructor. No parameters are hardcoded.

---

## Model Parameters

### `in_size`
The shape of a single input image as `[channels, height, width]`.
- `channels`: `3` for RGB images, `1` for grayscale
- Common sizes: `[3, 64, 64]`, `[3, 128, 128]`, `[3, 224, 224]`, `[3, 256, 256]`
- Larger sizes preserve more detail but require more memory and computation
- All images loaded from disk are automatically resized to match this

### `class_names`
An ordered list of class label strings. Each entry must exactly match the name of a subfolder in the dataset directory.
- Example: `["beer_lager", "beer_stout", "wine_red", "wine_white"]`
- The index of each name is the integer label the model outputs (e.g. `beer_lager = 0`, `beer_stout = 1`)
- Must be filled in before training — the list is empty by default

### `channels`
A list of integers defining the number of output channels for each convolutional layer. The length of this list determines the total number of conv layers.
- Use powers of 2: `16, 32, 64, 128, 256, 512`
- More channels = more capacity to learn complex patterns, but slower training and more memory
- A typical progression doubles the channels every few layers: `[32, 32, 64, 64, 128, 128]`
- Typical range per layer: `16–512`

### `pool_every`
How many convolutional layers to apply before each pooling step.
- Example: `2` means pool after every 2 conv layers
- Must be less than or equal to `len(channels)`
- Typical range: `1–4`
- More pooling = faster spatial reduction, smaller feature maps, less memory

### `hidden_dims`
A list of integers defining the output size of each fully-connected hidden layer in the classifier head. Sits between the flattened feature map and the final output layer.
- Use powers of 2: `64, 128, 256, 512, 1024`
- Fewer or smaller dims = less overfitting, faster inference
- Typical range per layer: `64–1024`
- Example: `[512, 256]` creates two hidden layers of size 512 and 256

### `activation`
The activation function applied after each convolutional and hidden linear layer.

| Value | Description |
|---|---|
| `relu` | Standard default. Fast and effective. Sets negative values to zero |
| `lrelu` | Leaky ReLU. Like relu but negative values get a small slope (0.01x). Reduces dead neurons |
| `tanh` | Squashes output to (-1, 1). Can work well in shallow networks |
| `sigmoid` | Squashes output to (0, 1). Rarely used in hidden layers |
| `softmax` | Converts scores to probabilities that sum to 1. Designed for output layers, not hidden layers |

`relu` is recommended for most cases.

### `use_batchnorm`
Whether to apply Batch Normalization after each convolutional layer.
- `true` or `false`
- Batch normalization normalizes the output of each layer, stabilizing and speeding up training
- Recommended for most cases, especially deeper networks

### `dropout_probability`
The probability of randomly zeroing out a neuron during training in the classifier head. Acts as a regularizer to reduce overfitting.
- `0.0` disables dropout entirely
- Typical range: `0.2–0.5`
- Higher values = stronger regularization. Too high can hurt accuracy

### `conv_kernel_size`
The size of the sliding window used in each convolutional layer. Must be an odd integer.
- `3` is the standard and most widely used choice
- `5` or `7` can capture wider spatial patterns but are slower
- Padding is automatically set to `kernel_size // 2` to preserve spatial dimensions

### `pooling_type`
The type of pooling applied after every `pool_every` conv layers.

| Value | Description |
|---|---|
| `max` | Takes the maximum value in each window. Standard for classification. Preserves dominant features |
| `avg` | Takes the average value in each window. Smoother but can lose sharp features |

### `pool_kernel_size`
The size of the pooling window.
- `2` halves the spatial dimensions at each pooling step (most common)
- `3` reduces more aggressively

### `image_normalization_mean`
Per-channel mean subtracted from each image during preprocessing. A 3-element list of floats.
- ImageNet values `[0.485, 0.456, 0.406]` work well for natural photos
- Use `[0.5, 0.5, 0.5]` as a simpler alternative

### `image_normalization_std`
Per-channel standard deviation used to scale each image during preprocessing. A 3-element list of floats.
- ImageNet values `[0.229, 0.224, 0.225]` work well for natural photos
- Use `[0.5, 0.5, 0.5]` as a simpler alternative

---

## Training Parameters

### `num_epochs`
The number of full passes over the training dataset.
- Typical range: `10–200`
- More epochs = more training time and risk of overfitting
- Recommended to monitor validation accuracy and stop when it plateaus

### `batch_size`
The number of images processed together in one forward/backward pass. Use powers of 2.
- Typical range: `8–128`
- Larger batches = faster training but more GPU memory required
- If training crashes with out-of-memory errors, reduce this value

### `learning_rate`
Controls how large each weight update step is during training.
- Typical range: `0.00001–0.01`
- Common starting points: `0.001` for Adam/AdamW, `0.01` for SGD
- Too high = unstable training (loss diverges). Too low = very slow convergence

### `num_dataloader_workers`
The number of background processes used to load images from disk in parallel.
- `0` loads images in the main process (safest, use this if you encounter errors on Windows)
- `4` is a good default on multi-core machines
- Range: `0–8`

### `optimizer`
The optimization algorithm used to update model weights during training.

| Value | Description |
|---|---|
| `Adam` | Adaptive learning rate. Safe default, works well out of the box |
| `AdamW` | Like Adam but with improved weight decay regularization. Often outperforms Adam |
| `SGD` | Stochastic Gradient Descent. Simpler, can generalize better with careful tuning |
| `RMSprop` | Adaptive, good for noisy or sparse gradients |

### `loss_function`
The function used to measure how wrong the model's predictions are during training.

| Value | Description |
|---|---|
| `CrossEntropyLoss` | Standard for multi-class classification. Internally applies softmax. Recommended |
| `NLLLoss` | Negative log-likelihood. Requires the model to output log-probabilities (pair with LogSoftmax) |

---

## Dataset Structure

Images must be organized as one subfolder per class:

```
dataset/
├── train/
│   ├── beer_lager/
│   │   ├── image_001.png
│   │   └── ...
│   ├── beer_stout/
│   │   └── ...
│   └── wine_red/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

The subfolder names must exactly match the entries in `class_names` in `config.json`.

---

## Augmentation

Offline augmentation classes are available in the `model/` directory. Run augmentation on the training set before training to expand the dataset. Each augmentation saves new image files alongside the originals.

| Class | File | Effect |
|---|---|---|
| `HorizontalFlipAugmentation` | `horizontal_flip_augmentation.py` | Mirrors images horizontally |
| `RotationAugmentation` | `rotation_augmentation.py` | Rotates by a random angle within a configured range |
| `ColorJitterAugmentation` | `color_jitter_augmentation.py` | Randomly adjusts brightness, contrast, and saturation |
