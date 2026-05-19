# Deepfake AI Model

Deepfake detection model using transfer learning (EfficientNet, XceptionNet, MobileNet).

## Project structure

```
deepfake-ai-model/
├── dataset/          # Dataset (not pushed to git)
│   ├── train/real/
│   ├── train/fake/
│   ├── val/real/
│   ├── val/fake/
│   ├── test/real/
│   └── test/fake/
├── models/           # Saved model files
├── results/          # Plots and metrics
├── src/              # Source code
│   ├── config.py            # Configuration
│   ├── prepare_dataset.py   # Dataset preparation
│   ├── model_builder.py     # Model architectures
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── predict.py           # Prediction script
│   └── utils.py             # Utility functions
├── notebooks/        # Jupyter notebooks
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

## Usage

1. Prepare dataset:
```bash
python src/prepare_dataset.py --zip_path ../archive\(1\).zip
```

2. Train model:
```bash
python src/train.py --model efficientnet
```

3. Evaluate:
```bash
python src/evaluate.py --model_path models/efficientnet_best.h5
```

4. Predict single image:
```bash
python src/predict.py --image_path /path/to/image.jpg
```
