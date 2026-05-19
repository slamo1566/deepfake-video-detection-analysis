"""
Model builder using transfer learning.

Supports:
    - EfficientNetB0  (default, lightweight, good accuracy)
    - Xception        (heavy, high accuracy)
    - MobileNet       (very lightweight)

Each model is built with:
    - Pre-trained ImageNet weights
    - Global Average Pooling
    - Dropout for regularization
    - Dense output layer with sigmoid (binary classification)
"""

from tensorflow.keras.applications import EfficientNetB0, Xception, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from src.config import IMG_SIZE, LEARNING_RATE, CLASS_LABELS


MODEL_REGISTRY = {
    "efficientnet": {
        "constructor": EfficientNetB0,
        "input_shape": (IMG_SIZE, IMG_SIZE, 3),
    },
    "xception": {
        "constructor": Xception,
        "input_shape": (IMG_SIZE, IMG_SIZE, 3),
    },
    "mobilenet": {
        "constructor": MobileNetV2,
        "input_shape": (IMG_SIZE, IMG_SIZE, 3),
    },
}


def build_model(model_name="efficientnet", trainable_base=False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[model_name]
    input_shape = entry["input_shape"]

    base_model = entry["constructor"](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = trainable_base

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"[+] Built {model_name} model (base trainable={trainable_base})")
    model.summary()

    return model
