import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_ZIP = BASE_DIR.parent / "archive(1).zip"
DATASET_EXTRACT_DIR = BASE_DIR / "dataset_extracted"

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

CLASS_LABELS = ["real", "fake"]

AVAILABLE_MODELS = ["efficientnet", "xception", "mobilenet"]
DEFAULT_MODEL = "efficientnet"

VIDEOS_DIR = DATASET_DIR / "videos"
VIDEOS_REAL_DIR = VIDEOS_DIR / "real"
VIDEOS_FAKE_DIR = VIDEOS_DIR / "fake"

FRAME_EXTRACT_RATE = 10

CHECKPOINT_PATH = MODELS_DIR / "{model_name}_best.h5"
FINAL_MODEL_PATH = MODELS_DIR / "{model_name}_final.h5"
HISTORY_PATH = RESULTS_DIR / "{model_name}_history.json"
EVALUATION_PATH = RESULTS_DIR / "{model_name}_evaluation.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "{model_name}_confusion_matrix.png"
CURVES_PATH = RESULTS_DIR / "{model_name}_curves.png"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "{model_name}_classification_report.txt"
