"""
Training script.

Usage:
    python src/train.py --model efficientnet
    python src/train.py --model xception
    python src/train.py --model mobilenet

What it does:
    1. Loads and preprocesses images from dataset/train/ and dataset/val/
    2. Builds the chosen model with transfer learning
    3. Trains with checkpoints and early stopping
    4. Saves training history as JSON
    5. Saves accuracy/loss curves as PNG
"""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    CHECKPOINT_PATH,
    FINAL_MODEL_PATH,
    HISTORY_PATH,
    CURVES_PATH,
)
from src.model_builder import build_model
from src.utils import save_history, plot_training_curves


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"[+] Training samples: {train_gen.samples}")
    print(f"[+] Validation samples: {val_gen.samples}")
    return train_gen, val_gen


def train(model_name):
    print(f"\n{'='*60}")
    print(f"  Training {model_name} model")
    print(f"{'='*60}\n")

    train_gen, val_gen = get_data_generators()
    model = build_model(model_name, trainable_base=False)

    checkpoint_path = str(CHECKPOINT_PATH).format(model_name=model_name)
    final_path = str(FINAL_MODEL_PATH).format(model_name=model_name)

    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(final_path)
    print(f"[+] Final model saved to {final_path}")

    history_path = str(HISTORY_PATH).format(model_name=model_name)
    save_history(history, history_path)

    curves_path = str(CURVES_PATH).format(model_name=model_name)
    plot_training_curves(history, curves_path)


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "xception", "mobilenet"],
        help="Model architecture to train",
    )
    args = parser.parse_args()
    train(args.model)


if __name__ == "__main__":
    main()
