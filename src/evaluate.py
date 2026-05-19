"""
Evaluation script.

Usage:
    python src/evaluate.py --model_path models/efficientnet_best.h5

What it does:
    1. Loads the test dataset
    2. Loads the trained model
    3. Computes accuracy, precision, recall, F1-score
    4. Generates confusion matrix
    5. Saves all metrics to JSON
    6. Prints classification report
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TEST_DIR, IMG_SIZE, BATCH_SIZE, EVALUATION_PATH, CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH
from src.utils import plot_confusion_matrix, save_classification_report, compute_metrics, save_evaluation


def evaluate(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_name = model_path.stem.replace("_best", "").replace("_final", "")
    print(f"[+] Loading model from {model_path}")
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"[+] Test samples: {test_gen.samples}")
    print("[+] Running evaluation ...")

    y_pred_prob = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    metrics = compute_metrics(y_true, y_pred)
    print(f"\n{'='*40}")
    print("  Evaluation Results")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    eval_path = str(EVALUATION_PATH).format(model_name=model_name)
    save_evaluation(metrics, eval_path)

    cm_path = str(CONFUSION_MATRIX_PATH).format(model_name=model_name)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    report_path = str(CLASSIFICATION_REPORT_PATH).format(model_name=model_name)
    save_classification_report(y_true, y_pred, report_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained .h5 model file",
    )
    args = parser.parse_args()
    evaluate(args.model_path)


if __name__ == "__main__":
    main()
