import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = "deepfake_detection_secret_key"

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
EXTRACTED_FRAMES_DIR = BASE_DIR / "extracted_frames"
CROPPED_FACES_DIR = BASE_DIR / "cropped_faces"
REPORTS_DIR = BASE_DIR / "reports"
TEMP_DIR = BASE_DIR / "temp"
STATIC_RESULTS_DIR = BASE_DIR / "static" / "results"
MODELS_DIR = BASE_DIR / "models"

HISTORY_FILE = REPORTS_DIR / "history.json"
MODEL_PATH = MODELS_DIR / "efficientnet_best.h5"

ALLOWED_EXTENSIONS = {"mp4", "avi"}
MAX_FILE_SIZE_MB = 500

PROJECT_FOLDERS = [
    UPLOAD_DIR,
    EXTRACTED_FRAMES_DIR,
    CROPPED_FACES_DIR,
    REPORTS_DIR,
    TEMP_DIR,
    STATIC_RESULTS_DIR,
    MODELS_DIR,
]

for folder in PROJECT_FOLDERS:
    folder.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


_ai_model = None
_ai_model_loaded = False
_ai_model_error = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_history(result: dict) -> None:
    history = load_history()
    history.insert(0, result)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


def find_result_by_id(analysis_id: str):
    history = load_history()

    for item in history:
        if item.get("analysis_id") == analysis_id:
            return item

    return None


def get_video_metadata(video_path: Path) -> dict:
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError("Vidéo corrompue ou illisible.")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None or fps <= 0:
        fps = 0
        duration = 0
    else:
        duration = frame_count / fps

    video.release()

    file_size = round(video_path.stat().st_size / (1024 * 1024), 2)

    return {
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "file_size": file_size,
    }


def extract_suspect_frames(video_path: Path, analysis_id: str, max_frames: int = 8) -> list:
    result_dir = STATIC_RESULTS_DIR / analysis_id
    result_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 40) if total_frames > 0 else 10

    frame_index = 0
    saved_count = 0
    suspect_frames = []

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_index % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

                output_name = f"frame_{saved_count + 1:03d}.jpg"
                output_path = result_dir / output_name

                cv2.imwrite(str(output_path), frame)

                suspect_frames.append(
                    {
                        "frame_number": frame_index,
                        "image_path": f"results/{analysis_id}/{output_name}",
                        "score": None,
                    }
                )

                saved_count += 1

                if saved_count >= max_frames:
                    break

        frame_index += 1

    video.release()
    return suspect_frames


def load_ai_model():
    global _ai_model, _ai_model_loaded, _ai_model_error

    if _ai_model_loaded:
        return _ai_model

    _ai_model_loaded = True

    if not MODEL_PATH.exists():
        _ai_model_error = "Modèle IA introuvable. Mode démonstration activé."
        return None

    try:
        from src.predict import load_trained_model

        _ai_model = load_trained_model(str(MODEL_PATH))
        _ai_model_error = None
        return _ai_model

    except Exception as error:
        _ai_model_error = f"Erreur chargement modèle IA : {error}"
        return None


def predict_with_ai(suspect_frames: list) -> tuple:
    model = load_ai_model()

    if model is None:
        return [], "DEMO"

    try:
        from src.predict import predict_image

        predictions = []

        for frame in suspect_frames:
            image_absolute_path = BASE_DIR / "static" / frame["image_path"]
            prediction = predict_image(model, str(image_absolute_path))

            label = prediction["prediction"].upper()
            confidence = float(prediction["confidence"])

            if label == "FAKE":
                fake_probability = confidence
            else:
                fake_probability = 1 - confidence

            predictions.append(fake_probability)
            frame["score"] = round(fake_probability * 100, 2)

        return predictions, "AI"

    except Exception:
        return [], "DEMO"


def demo_predictions(suspect_frames: list, analysis_id: str) -> list:
    seed = sum(ord(char) for char in analysis_id)
    rng = np.random.default_rng(seed)

    predictions = []

    for frame in suspect_frames:
        fake_probability = float(rng.uniform(0.35, 0.92))
        frame["score"] = round(fake_probability * 100, 2)
        predictions.append(fake_probability)

    return predictions


def analyze_video(video_path: Path, original_filename: str, analysis_id: str) -> dict:
    metadata = get_video_metadata(video_path)
    suspect_frames = extract_suspect_frames(video_path, analysis_id)

    frame_predictions, mode = predict_with_ai(suspect_frames)

    if not frame_predictions:
        frame_predictions = demo_predictions(suspect_frames, analysis_id)
        mode = "DEMO"

    if frame_predictions:
        fake_probability = float(np.mean(frame_predictions)) * 100
        real_probability = 100 - fake_probability

        if fake_probability >= 50:
            final_prediction = "FAKE"
            confidence = fake_probability
        else:
            final_prediction = "REAL"
            confidence = real_probability
    else:
        fake_probability = 0
        real_probability = 0
        final_prediction = "NON ANALYSABLE"
        confidence = 0

    result = {
        "analysis_id": analysis_id,
        "video_name": original_filename,
        "stored_video": video_path.name,
        "prediction": final_prediction,
        "confidence": round(confidence, 2),
        "real_probability": round(real_probability, 2),
        "fake_probability": round(fake_probability, 2),
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata,
        "suspect_frames": suspect_frames,
        "mode": mode,
        "model_message": _ai_model_error,
    }

    save_history(result)
    return result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        flash("Aucun fichier envoyé.", "danger")
        return redirect(url_for("index"))

    file = request.files["video"]

    if file.filename == "":
        flash("Aucune vidéo sélectionnée.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format non accepté. Utilisez uniquement MP4 ou AVI.", "danger")
        return redirect(url_for("index"))

    analysis_id = str(uuid.uuid4())
    original_filename = secure_filename(file.filename)
    extension = original_filename.rsplit(".", 1)[1].lower()
    stored_filename = f"{analysis_id}.{extension}"
    video_path = UPLOAD_DIR / stored_filename

    file.save(video_path)

    try:
        analyze_video(video_path, original_filename, analysis_id)
        return redirect(url_for("result", analysis_id=analysis_id))

    except ValueError as error:
        flash(str(error), "danger")
        return redirect(url_for("index"))

    except Exception as error:
        flash(f"Erreur pendant l'analyse : {error}", "danger")
        return redirect(url_for("index"))


@app.route("/result/<analysis_id>", methods=["GET"])
def result(analysis_id):
    analysis_result = find_result_by_id(analysis_id)

    if analysis_result is None:
        flash("Résultat introuvable.", "warning")
        return redirect(url_for("history"))

    return render_template("result.html", result=analysis_result)


@app.route("/history", methods=["GET"])
def history():
    analyses = load_history()
    return render_template("history.html", analyses=analyses)


@app.errorhandler(413)
def file_too_large(error):
    flash(f"Fichier trop volumineux. Taille maximale : {MAX_FILE_SIZE_MB} MB.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)