# Beer & Wine Image Classifier

A convolutional neural network for classifying types of beer and wine from images.

---

## Setup

```bash
pip install -r requirements.txt
```

If you're running the YOLO preprocessing pipeline, you'll also need:

```bash
pip install ultralytics opencv-python imagehash pillow-heif
```

---

## Data Preparation

Start by collecting raw images, one subfolder per class:

```
raw_images/
├── Cabernet Sauvignon/
├── Chardonnay/
├── IPA/
└── ...
```

Subfolder names must exactly match the `class_names` list in `config.json`.

After preprocessing (see below), the dataset needs to be split into `train/`, `val/`, and `test/` before training can begin:

```
dataset/
├── train/
│   ├── Cabernet Sauvignon/
│   ├── Chardonnay/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

## YOLO Preprocessing Pipeline

Raw images contain backgrounds, cluttered scenes, and irrelevant objects. This pipeline detects the glass or bottle in each image using YOLO, crops it out, and leaves you with a clean dataset. Run these steps in order from the project root.

---

### Step 1 — Convert to PNG

If your images are in mixed formats (HEIC, WebP, AVIF, etc.), normalize them all to PNG first:

```bash
python utilities/convert_to_png.py <path/to/raw_images> -r
```

The originals are deleted after a successful conversion. Use `-o <output_dir>` to write to a separate folder instead.

---

### Step 2 — Remove Duplicates

Before running detection, remove near-duplicate images that would contaminate train/test splits:

```bash
python utilities/duplicate_finder.py <path/to/raw_images> -r
```

A timestamped report is saved listing all duplicate groups. Review it and delete the ones you don't want to keep before moving on.

---

### Step 3 — Detect Objects

Run YOLO over the dataset to locate the objects of interest in each image:

```bash
python utilities/yolo_detector.py <path/to/raw_images> \
    --model yolo11n.pt \
    --confidence 0.25 \
    --target-classes cup "wine glass" \
    --recursive
```

The model weights download automatically on first run. The output is a `detections.json` file mapping each image to its detected bounding boxes.

---

### Step 4 — Verify Detections (Optional)

Draw the bounding boxes on the images and take a look before cropping:

```bash
python utilities/yolo_visualizer.py <path/to/detections.json> \
    --output-folder <path/to/visualized>
```

If detections look off, go back to Step 3 and adjust `--confidence` or `--target-classes`.

---

### Step 5 — Crop

Extract each detected region from the original images and group them by class:

```bash
python utilities/crop_yolo.py <path/to/raw_images> <path/to/detections.json> \
    --confidence-threshold 0.5
```

The output lands in a `cropped images/` subfolder inside the input folder, with one subfolder per detected class name.

---

### Step 6 — Rename

Move the `cropped images/` folder into `dataset/`, then run:

```bash
python utilities/rename_images.py
```

This standardizes filenames to `{class_name}{index}.png` across the whole cropped dataset.

---

### Step 7 — Split

Split the cropped images into train/val/test:

```bash
python utilities/dataset_splitter.py
```

Open the script first and set the split ratios and source/destination paths to match your setup. The output is the `dataset/` structure training expects.

---

## Running Training

Once the dataset is in place and `config.json` is set up:

```bash
python main.py
```

The script validates the dataset structure, builds the model, runs training (with cross-validation if enabled), and saves results to a new versioned folder under `results/`. To resume from a checkpoint, set `resume_from_checkpoint` in `config.json` to the path of an existing `checkpoint.pth`.

### Submitting to a SLURM Cluster

Two scripts are provided for submitting jobs via SLURM.

**`run.sh`** — trains the model on a GPU node:

```bash
sbatch run.sh
```

Requests 1 GPU, 16 GB RAM, and up to 24 hours. After training completes, it automatically generates training curve plots and confusion matrices, then moves everything (including the SLURM logs) into the versioned results folder.

**`run_confusion_matrices.sh`** — regenerates confusion matrices for any results folder that's missing them, without re-running training:

```bash
sbatch run_confusion_matrices.sh
```

Requests 2 CPUs and 8 GB RAM (no GPU needed). It scans every `results/v*/` folder and runs the confusion matrix generator on any that have a `results.json` but no plots yet.

---

## Configuration

Everything is controlled through `config.json`. The key sections are:

- **Model** — input size, class names, conv channels, pooling, hidden layers, activation, batchnorm, dropout
- **Training** — epochs, batch size, learning rate, weight decay, optimizer, loss function, class weights, label smoothing, early stopping, dataloader workers
- **Augmentations** — horizontal flip, rotation, gaussian blur, perspective transform (each with `enabled` and `probability` flags)
- **Cross-validation** — enable K-fold CV and set the number of folds
- **Pretrained model** — swap the custom CNN for a pretrained backbone (e.g. ResNet50) and optionally freeze it
