# Analysis and Detection of Deepfakes in Videography
*(Analyse et Détection des Deepfakes en Vidéographie)*

This repository contains an automated system designed to detect and analyze Deepfakes in video content. By leveraging Artificial Intelligence and Deep Learning, the application processes input videos to identify facial manipulation anomalies and classify them as **REAL** or **FAKE**, providing an associated confidence score.

> ⚠️ **Note:** This project focuses *exclusively* on deepfake detection and analysis. It does not include tools for generating or creating synthetic media.

---

## 📌 Context & Problem Statement
The rapid evolution of Deep Learning has enabled the creation of highly realistic Deepfakes using Generative Adversarial Networks (GANs), Auto-Encoders, and other image synthesis architectures. These manipulations have reached a level of realism where the human eye can no longer easily distinguish between authentic and falsified media.

This project addresses this issue by using computer vision and convolutional neural networks (CNNs) to automatically detect specific facial and structural anomalies, including:
* Face texture inconsistencies
* Blending defects and visual artifacts
* Anomalies around the eyes and mouth
* Lighting inconsistencies
* Synthesis imperfections generated during face creation

---

## 🏗️ System Architecture & Workflow
The core of the system is a structured data processing pipeline that transforms raw video into individual face evaluations:

```
Video Input ──> Frame Extraction ──> Face Detection & Cropping ──> Data Preprocessing ──> AI Model (CNN) ──> Classification
```

### Detailed Workflow Steps

1. **Video Ingestion:** Supports standard video formats like `.mp4`, `.avi`, and other compatible files.
2. **Frame Extraction:** Converts videos into a series of static images (e.g., extracting one frame every 10 frames) to optimize performance and reduce computational complexity.
3. **Face Detection & Extraction:** Isolates and crops the human face from each frame using detectors like MTCNN or Dlib to focus the analysis exclusively on the modified regions.
4. **Data Preprocessing:** Standardizes the isolated face images to ensure model compatibility. This includes:
   * Resizing images to a fixed 224x224 resolution.
   * Pixel normalization.
   * Face alignment and quality enhancement.
5. **AI Analysis & Classification:** Preprocessed faces are fed into deep learning models (such as **EfficientNet** or **XceptionNet**) utilizing **Transfer Learning**. This pre-trained model approach ensures higher accuracy, faster training times, and lower GPU resource requirements.
6. **Final Output:** The system aggregates individual frame analyses to display a final classification alongside a certainty percentage:
   * *Example:* `Prediction: FAKE - Confidence: 92%`
   * *Example:* `Prediction: REAL - Confidence: 87%`

---

## 🛠️ Technologies & Tools
The project is built using a modern Python-based ecosystem:

* **Core Language:** Python
* **Video & Image Processing:** OpenCV, NumPy, Pandas
* **Face Detection:** MTCNN, Dlib
* **Deep Learning Frameworks:** TensorFlow / Keras, PyTorch
* **User Interface:** Flask (Web Graphical Interface)

---

## 📊 Dataset
The model is trained and evaluated using the following benchmark dataset:

* **[Deep Fake Detection (DFD) Entire Original Dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset)** (via Kaggle)
  * **Description:** A comprehensive video-based dataset downloaded from the official FaceForensics server.
  * **Size:** ~3,400 video files in standard MP4 format.
  * **Content:** Covers a balanced collection of both original and manipulated video sequences (REAL vs. FAKE annotations), specifically optimized for evaluating facial manipulation algorithms.

---

## 🤖 AI Model — Member 4

### Project structure

```
deepfake-ai-model/
├── dataset/                   # Dataset (not pushed to git)
│   ├── videos/real/           # Original DFD videos
│   ├── videos/fake/           # Manipulated DFD videos
│   ├── train/real/            # Training face images (real)
│   ├── train/fake/            # Training face images (fake)
│   ├── val/real/              # Validation face images
│   ├── val/fake/              # Validation face images
│   ├── test/real/             # Test face images
│   └── test/fake/             # Test face images
├── models/                    # Saved .h5 model files
├── results/                   # Plots, metrics, classification reports
├── src/                       # Source code
│   ├── config.py              # Paths & hyperparameters
│   ├── prepare_dataset.py     # Extract zip, organize videos, extract frames & faces
│   ├── model_builder.py       # EfficientNet / Xception / MobileNet builder
│   ├── train.py               # Training loop with callbacks
│   ├── evaluate.py            # Accuracy, precision, recall, F1, confusion matrix
│   ├── predict.py             # Single image → {prediction, confidence}
│   └── utils.py               # Plotting & metrics helpers
├── notebooks/                 # Jupyter notebooks
├── requirements.txt
├── .gitignore
└── README.md
```

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

**1. Prepare dataset** (organize videos from the DFD zip):
```bash
python src/prepare_dataset.py --zip_path /path/to/archive.zip
```

**2. Train model** (after face images are in `dataset/train/val/test`):
```bash
python src/train.py --model efficientnet
```

**3. Evaluate model**:
```bash
python src/evaluate.py --model_path models/efficientnet_best.h5
```

**4. Predict single image**:
```bash
python src/predict.py --image_path /path/to/face.jpg
```

### Flask integration

```python
from src.predict import load_trained_model, predict_image

model = load_trained_model("models/efficientnet_best.h5")
result = predict_image(model, "uploads/face.jpg")
# result == {"prediction": "fake", "confidence": 0.9876}
```
