"""
Step 2 of pipeline: benchmark all 10 pretrained models on saved tensors.

Run after preprocess.py:
    python benchmark.py

Prints accuracy table for all 10 models and saves results to benchmark_results.txt
"""

import os, glob, torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 20
MODELS_DIR = Path("models/Models")
FAKE_DIR   = Path("preprocessed/fake")
REAL_DIR   = Path("preprocessed/real")


# ── Model architecture ────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048):
        super().__init__()
        base = models.resnext50_32x4d(pretrained=False)
        self.model   = nn.Sequential(*list(base.children())[:-2])
        self.lstm    = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bias=False)
        self.dp      = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x      = x.view(b * t, c, h, w)
        fmap   = self.model(x)
        x      = self.avgpool(fmap).view(b, t, 2048)
        x, _   = self.lstm(x, None)
        return self.dp(self.linear1(x[:, -1, :]))


sm = nn.Softmax(dim=1)


def load_tensors(folder):
    tensors, names = [], []
    for pt in sorted(folder.glob("*.pt")):
        t = torch.load(pt, map_location="cpu")
        # ensure shape [SEQ_LEN, C, H, W]
        tensors.append(t[:SEQ_LEN].unsqueeze(0))
        names.append(pt.stem)
    return tensors, names


def evaluate_model(model_path, fake_tensors, real_tensors):
    net = Model().to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    correct = total = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for tensor in fake_tensors:
            logits = net(tensor.to(DEVICE))
            pred   = int(torch.argmax(sm(logits), dim=1).item())
            # 0=FAKE, 1=REAL
            if pred == 0: tp += 1
            else:         fn += 1
            total += 1

        for tensor in real_tensors:
            logits = net(tensor.to(DEVICE))
            pred   = int(torch.argmax(sm(logits), dim=1).item())
            if pred == 1: tn += 1
            else:         fp += 1
            total += 1

    correct    = tp + tn
    accuracy   = round(100 * correct / total, 1)
    precision  = round(100 * tp / (tp + fp), 1) if (tp + fp) > 0 else 0
    recall     = round(100 * tp / (tp + fn), 1) if (tp + fn) > 0 else 0

    return accuracy, precision, recall, correct, total


def main():
    print(f"Device: {DEVICE}")
    print("Loading tensors...")
    fake_tensors, _ = load_tensors(FAKE_DIR)
    real_tensors, _ = load_tensors(REAL_DIR)
    print(f"Fake tensors: {len(fake_tensors)}  |  Real tensors: {len(real_tensors)}")

    model_files = sorted(MODELS_DIR.glob("*.pt"))
    print(f"\nBenchmarking {len(model_files)} models on {len(fake_tensors)+len(real_tensors)} videos...\n")

    header = f"{'Model':<45} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'Correct':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for mpath in model_files:
        acc, prec, rec, correct, total = evaluate_model(mpath, fake_tensors, real_tensors)
        name = mpath.name
        print(f"{name:<45} {acc:>8.1f}% {prec:>9.1f}% {rec:>7.1f}% {correct:>5}/{total}")
        results.append((name, acc, prec, rec, correct, total))

    # save to file
    out = Path("benchmark_results.txt")
    with open(out, "w") as f:
        f.write(header + "\n" + "-" * len(header) + "\n")
        for name, acc, prec, rec, correct, total in results:
            f.write(f"{name:<45} {acc:>8.1f}% {prec:>9.1f}% {rec:>7.1f}% {correct:>5}/{total}\n")

    best = max(results, key=lambda x: x[1])
    print(f"\n✓ Best model: {best[0]}  →  {best[1]}% accuracy")
    print(f"✓ Results saved to {out}")


if __name__ == "__main__":
    main()
