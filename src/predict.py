"""
Prediction script.

Usage:
    python src/predict.py --image_path /path/to/image.jpg

Output:
    {"prediction": "fake", "confidence": 0.9876}
    {"prediction": "real", "confidence": 0.9987}

This script is designed to be easily integrated with Flask (member 5).
Just call predict_image() from your Flask route.
"""

import argparse
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from src.config import IMG_SIZE, CLASS_LABELS


def load_trained_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image_path):
    image_array = preprocess_image(image_path)
    prob = model.predict(image_array, verbose=0)[0][0]

    if prob >= 0.5:
        prediction = "fake"
        confidence = float(prob)
    else:
        prediction = "real"
        confidence = float(1 - prob)

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/efficientnet_best.h5",
        help="Path to trained model",
    )
    args = parser.parse_args()

    model = load_trained_model(args.model_path)
    result = predict_image(model, args.image_path)
    print(result)


if __name__ == "__main__":
    main()
