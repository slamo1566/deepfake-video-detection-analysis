# Deepfake Video Detection
*Analysis and Detection of Deepfakes in Videography*

A web application that detects deepfake videos using deep learning. Upload a video and the system automatically extracts frames, detects faces, and classifies the video as **REAL** or **FAKE** with a confidence score.

---

## How It Works

The detection pipeline has 3 stages:

```
Video ──► Frame Extraction ──► Face Detection ──► AI Model ──► REAL / FAKE + Confidence
```

1. **Frame Extraction** — Samples up to 20 frames evenly spaced across the video using OpenCV.
2. **Face Detection** — Detects and crops the face region from each frame using the `face_recognition` library (HOG-based detector).
3. **AI Classification** — A pretrained **ResNeXt-50 + LSTM** model analyzes the sequence of face crops and outputs a final prediction with confidence percentage.

The LSTM processes frames as a sequence, capturing temporal inconsistencies that single-frame models would miss. This is key for detecting subtle deepfake artifacts that appear across multiple frames.

---

## Project Structure

```
deepfake-ai-model/
│
├── app.py                    # Flask web application — entry point
├── benchmark.py              # Benchmarks all 10 pretrained models on a test set
├── preprocess.py             # Preprocesses videos into face tensors (run before benchmark)
│
├── src/
│   ├── predict_video.py      # Core prediction pipeline (video → REAL/FAKE)
│   └── __init__.py
│
├── templates/
│   ├── index.html            # Upload page
│   └── success.html          # Results page (shows prediction + confidence)
│
├── models/
│   └── Models/               # Pretrained .pt model files (download separately — see below)
│
├── test_videos/              # 4 sample videos for demonstration
│   ├── clearly_fake.mp4      # Deepfake detected with 100% confidence
│   ├── subtle_fake.mp4       # Deepfake detected with ~50% confidence (hard case)
│   ├── clearly_real.mp4      # Authentic video detected with 100% confidence
│   └── subtle_real.mp4       # Authentic video detected with ~62% confidence (hard case)
│
├── uploads/                  # Videos uploaded via the web interface (auto-created)
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Flask server — handles video upload, calls the prediction pipeline, renders results |
| `src/predict_video.py` | The core module — loads the model, extracts frames, detects faces, runs inference |
| `benchmark.py` | Evaluates all 10 pretrained models on a preprocessed dataset and prints an accuracy table |
| `preprocess.py` | Reads videos, extracts face sequences, saves them as PyTorch tensors for fast benchmarking |
| `templates/index.html` | Upload form where the user submits a video |
| `templates/success.html` | Results page showing REAL/FAKE verdict, confidence bar, and video metadata |

---

## Model

**Architecture:** ResNeXt-50 (CNN backbone) + LSTM (temporal sequence modeling)

- The **ResNeXt-50** extracts a 2048-dimensional feature vector from each face crop
- The **LSTM** processes the sequence of frame features and captures temporal patterns
- Final **linear classifier** outputs REAL or FAKE probabilities

**Pretrained model used:** `model_87_acc_20_frames_final_data.pt`
- Trained on the FaceForensics++ dataset
- 20 frames per video
- **87% accuracy** on the original benchmark

### Benchmark Results (on 200 FaceForensics++ videos)

| Model | Frames | Accuracy | Precision | Recall | Correct |
|---|---|---|---|---|---|
| model_90_acc_60_frames_final_data | 60 | **63.5%** | 70.1% | 47.0% | 127/200 |
| **model_87_acc_20_frames_final_data** ✅ | **20** | 61.0% | 58.2% | **78.0%** | 122/200 |
| model_93_acc_100_frames_celeb_FF_data | 100 | 57.0% | 58.5% | 48.0% | 114/200 |
| model_84_acc_10_frames_final_data | 10 | 56.5% | 54.1% | 85.0% | 113/200 |
| model_89_acc_40_frames_final_data | 40 | 54.5% | 53.0% | 79.0% | 109/200 |
| model_97_acc_80_frames_FF_data | 80 | 52.0% | 83.3% | 5.0% | 104/200 |
| model_97_acc_60_frames_FF_data | 60 | 51.0% | 100.0% | 2.0% | 102/200 |
| model_97_acc_100_frames_FF_data | 100 | 50.5% | 66.7% | 2.0% | 101/200 |
| model_95_acc_40_frames_FF_data | 40 | 49.0% | 25.0% | 1.0% | 98/200 |
| model_90_acc_20_frames_FF_data | 20 | 36.5% | 21.3% | 10.0% | 73/200 |

> **Note:** Models suffixed `_FF_data` were trained exclusively on FaceForensics++ and perform poorly on DeepFakeDetection-style videos (low recall). Models suffixed `_final_data` generalize better across manipulation methods.

We chose `model_87` for its **best balance of recall (78%) and speed (20 frames)** — it catches the most fake videos while being fast enough for real-time demo use.

### Download the Model

The pretrained models are too large for GitHub (217MB each). Download them from:

**[Google Drive — All Models](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-)**

Place the downloaded `.pt` files in:
```
models/Models/
```

---

## Installation

```bash
git clone https://github.com/slamo1566/deepfake-video-detection-analysis.git
cd deepfake-video-detection-analysis

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the model (see above) and place it in `models/Models/`.

---

## Running the App

```bash
source venv/bin/activate
python app.py
```

Open your browser at **http://127.0.0.1:5000**, upload a video (`.mp4` or `.avi`), and the result appears automatically.

---

## Flask Integration (for developers)

```python
from src.predict_video import load_model, predict_video

model = load_model("models/Models/model_87_acc_20_frames_final_data.pt", sequence_length=20)
result = predict_video(model, "path/to/video.mp4", sequence_length=20)
# {"prediction": "FAKE", "confidence": 91.3, "frames_analyzed": 20}
```

---

## Dataset

Evaluated on **FaceForensics++** (C23 compressed), which contains:
- 1,000 original authentic videos
- 1,000+ manipulated videos (DeepFakeDetection method)

**Source:** [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)

---

## Technologies

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, ResNeXt-50, LSTM |
| Face Detection | face_recognition (HOG + SVM) |
| Video Processing | OpenCV |
| Web Interface | Flask, Bootstrap 5 |
| GPU Acceleration | CUDA (NVIDIA RTX 4060) |

---

