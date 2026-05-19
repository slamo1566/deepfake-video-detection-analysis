import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.config import RESULTS_DIR, CLASS_LABELS


def save_history(history, path):
    history_dict = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    with open(path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[+] Training history saved to {path}")


def plot_training_curves(history, save_path):
    epochs = range(1, len(history.history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history.history["accuracy"], "b-", label="Train Accuracy")
    ax1.plot(epochs, history.history["val_accuracy"], "r-", label="Val Accuracy")
    ax1.set_title("Accuracy over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history.history["loss"], "b-", label="Train Loss")
    ax2.plot(epochs, history.history["val_loss"], "r-", label="Val Loss")
    ax2.set_title("Loss over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Confusion matrix saved to {save_path}")
    return cm


def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"[+] Classification report saved to {save_path}")
    print("\n" + report)


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary")),
        "recall": float(recall_score(y_true, y_pred, average="binary")),
        "f1_score": float(f1_score(y_true, y_pred, average="binary")),
    }


def save_evaluation(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Evaluation metrics saved to {path}")
