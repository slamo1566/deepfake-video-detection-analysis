# Deepfake AI Model

Deepfake detection model using transfer learning (EfficientNet, XceptionNet, MobileNet).
Trained on face images from the **Deep Fake Detection (DFD)** dataset.

## Project structure

```
deepfake-ai-model/
├── dataset/                   # Dataset (ignored by git)
│   ├── videos/real/           # Original videos
│   ├── videos/fake/           # Manipulated videos
│   ├── frames/real/           # Extracted video frames (optional)
│   ├── frames/fake/           # Extracted video frames (optional)
│   ├── faces/real/            # Cropped face images (optional)
│   ├── faces/fake/            # Cropped face images (optional)
│   ├── train/real/            # Training images (real)
│   ├── train/fake/            # Training images (fake)
│   ├── val/real/              # Validation images
│   ├── val/fake/              # Validation images
│   ├── test/real/             # Test images
│   └── test/fake/             # Test images
├── models/                    # Saved .h5 model files
├── results/                   # Plots, metrics, classification reports
├── src/                       # Source code
│   ├── config.py              # Paths & hyperparameters
│   ├── prepare_dataset.py     # Extract zip, organize, optionally extract frames & faces
│   ├── model_builder.py       # EfficientNet / Xception / MobileNet
│   ├── train.py               # Training loop with callbacks
│   ├── evaluate.py            # Accuracy, precision, recall, F1, confusion matrix
│   ├── predict.py             # Single image prediction → {prediction, confidence}
│   └── utils.py               # Plotting & metrics helpers
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

The DFD dataset contains **videos** (MP4), not images. You have two paths:

### Path A: Automatic (do it all yourself)

```bash
# Step 1: Extract, organize videos, extract frames, detect & crop faces
python src/prepare_dataset.py --zip_path /path/to/archive.zip --extract-frames --detect-faces

# Step 2: Train model
python src/train.py --model efficientnet
```

### Path B: Collaborative (wait for members 2 & 3)

```bash
# Step 1: Just extract and organize videos
python src/prepare_dataset.py --zip_path /path/to/archive.zip

# Step 2: Members 2 & 3 process videos → provide face images in:
#   dataset/train/real/  dataset/train/fake/
#   dataset/val/real/    dataset/val/fake/
#   dataset/test/real/   dataset/test/fake/

# Step 3: Train
python src/train.py --model efficientnet
```

## Training

```bash
# Train EfficientNet (default, lightweight)
python src/train.py --model efficientnet

# Train Xception (heavier, higher accuracy)
python src/train.py --model xception

# Train MobileNet (lightest)
python src/train.py --model mobilenet
```

## Evaluation

```bash
python src/evaluate.py --model_path models/efficientnet_best.h5
```

## Prediction (single image)

```bash
python src/predict.py --image_path /path/to/face.jpg --model_path models/efficientnet_best.h5
```

Output: `{"prediction": "fake", "confidence": 0.9876}`

## Flask integration (member 5)

```python
from src.predict import load_trained_model, predict_image

model = load_trained_model("models/efficientnet_best.h5")
result = predict_image(model, "uploads/face.jpg")
# result == {"prediction": "fake", "confidence": 0.9876}
```
