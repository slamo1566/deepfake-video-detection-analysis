"""
Test pipeline on 2 fake + 2 real FaceForensics++ videos.

Put videos in:
    dataset/videos/fake/   (2 fake videos)
    dataset/videos/real/   (2 real videos)

Then run:
    python test_pipeline.py
"""

import os
import cv2
import face_recognition
import torch
from pathlib import Path
from src.predict_video import load_model, predict_video

MODEL_PATH = "models/Models/model_90_acc_20_frames_FF_data.pt"
SEQ_LEN    = 20

FAKE_DIR = "dataset/videos/fake"
REAL_DIR = "dataset/videos/real"


def process_video(video_path, label, model):
    print(f"\n{'='*55}")
    print(f"  Video : {Path(video_path).name}")
    print(f"  Label : {label}")
    print(f"{'='*55}")

    # --- Step 1: Frame extraction ---
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"[1] Frame extraction  : {total_frames} total frames  |  {fps:.1f} fps")
    print(f"    Will sample       : up to {SEQ_LEN} frames with face")

    # --- Step 2 + 3: Face detection + Model classification ---
    print(f"[2] Face detection    : running face_recognition on each frame...")
    print(f"[3] Classification    : feeding face crops to ResNeXt-LSTM model...")

    result = predict_video(model, video_path, sequence_length=SEQ_LEN)

    print(f"\n  ✔ Frames with face analyzed : {result['frames_analyzed']}")
    print(f"  ✔ Prediction                : {result['prediction']}")
    print(f"  ✔ Confidence                : {result['confidence']}%")

    correct = (result['prediction'] == label)
    status  = "CORRECT ✓" if correct else "WRONG ✗"
    print(f"  ✔ Result                    : {status}")
    return result, correct


def main():
    print("\nLoading model...")
    model = load_model(MODEL_PATH, sequence_length=SEQ_LEN)

    results = []

    for label, folder in [("FAKE", FAKE_DIR), ("REAL", REAL_DIR)]:
        videos = [f for f in sorted(os.listdir(folder)) if f.endswith(('.mp4', '.avi'))][:2]
        if not videos:
            print(f"\n[!] No videos found in {folder}")
            continue
        for v in videos:
            r, correct = process_video(os.path.join(folder, v), label, model)
            results.append((v, label, r['prediction'], r['confidence'], correct))

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    for name, true_label, pred, conf, correct in results:
        status = "✓" if correct else "✗"
        print(f"  {status}  {name[:35]:35s}  true={true_label:4s}  pred={pred:4s}  conf={conf}%")

    correct_count = sum(1 for *_, c in results if c)
    print(f"\n  Accuracy: {correct_count}/{len(results)} correct")


if __name__ == "__main__":
    main()
