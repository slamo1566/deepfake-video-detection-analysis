"""
Video prediction using pretrained ResNeXt-50 + LSTM model.
For Member 5 (Flask integration).

Usage:
    python src/predict_video.py --video_path /path/to/video.mp4

Output:
    {"prediction": "FAKE", "confidence": 91.3, "frames_analyzed": 20}

Flask integration:
    from src.predict_video import load_model, predict_video

    model = load_model("models/Models/model_87_acc_20_frames_final_data.pt", sequence_length=20)
    result = predict_video(model, "uploads/video.mp4", sequence_length=20)
    # {"prediction": "FAKE", "confidence": 91.3, "frames_analyzed": 20}
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import face_recognition
from torchvision import transforms, models
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Model architecture (from backup repo) ──────────────────────────────────────
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

sm = nn.Softmax(dim=1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super().__init__()
        base = models.resnext50_32x4d(pretrained=False)
        self.model   = nn.Sequential(*list(base.children())[:-2])
        self.lstm    = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, bias=False)
        self.relu    = nn.LeakyReLU()
        self.dp      = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


# ── Public API ─────────────────────────────────────────────────────────────────

def load_model(model_path, sequence_length=20):
    """Load the pretrained model. Call once at app startup."""
    net = _Model(num_classes=2).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()
    print(f"[+] Loaded model: {model_path}  |  device: {DEVICE}")
    return net


def predict_video(model, video_path, sequence_length=20):
    """
    Classify a video as REAL or FAKE.

    Args:
        model: loaded model from load_model()
        video_path: path to video file (mp4, avi, etc.)
        sequence_length: number of frames to sample (default 20)

    Returns:
        dict: {"prediction": "REAL"/"FAKE", "confidence": float (0-100), "frames_analyzed": int}
    """
    frames = _extract_frames(video_path, sequence_length)

    if len(frames) == 0:
        return {"prediction": "unknown", "confidence": 0.0, "frames_analyzed": 0}

    # pad if video shorter than sequence_length
    while len(frames) < sequence_length:
        frames.append(frames[-1])

    tensor = torch.stack(frames[:sequence_length]).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]

    with torch.no_grad():
        _, logits = model(tensor)
        probs = sm(logits)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = round(probs[0, pred_idx].item() * 100, 1)

    # 0 = FAKE, 1 = REAL  (same convention as original repo)
    prediction = "REAL" if pred_idx == 1 else "FAKE"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "frames_analyzed": len(frames),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_frames(video_path, sequence_length):
    """Extract up to sequence_length face-cropped frames from the video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    padding = 40

    while cap.isOpened() and len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)

        if face_locs:
            top, right, bottom, left = face_locs[0]
            top    = max(0, top - padding)
            bottom = min(rgb.shape[0], bottom + padding)
            left   = max(0, left - padding)
            right  = min(rgb.shape[1], right + padding)
            face = rgb[top:bottom, left:right]
        else:
            face = rgb  # no face found — use full frame

        frames.append(transform(face))

    cap.release()
    return frames


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",  required=True)
    parser.add_argument("--model_path",  default="models/Models/model_87_acc_20_frames_final_data.pt")
    parser.add_argument("--seq_len",     type=int, default=20)
    args = parser.parse_args()

    model  = load_model(args.model_path, args.seq_len)
    result = predict_video(model, args.video_path, args.seq_len)
    print(result)


if __name__ == "__main__":
    main()
