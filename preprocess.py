"""
Step 1 of pipeline: videos → frames → face detection → save tensors

Run once:
    python preprocess.py

Output: preprocessed/fake/*.pt  and  preprocessed/real/*.pt
Each .pt file = face sequence tensor [seq_len, C, H, W] for one video.
"""

import os, cv2, torch, face_recognition
from pathlib import Path
from torchvision import transforms

SEQ_LEN  = 20     # frames per video
IM_SIZE  = 112
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
PADDING  = 40

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

FAKE_DIR = Path("dataset/videos/fake")
REAL_DIR = Path("dataset/videos/real")
OUT_DIR  = Path("preprocessed")


def process_video(video_path, out_path):
    if out_path.exists():
        return "skip"

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // SEQ_LEN)

    frames = []
    idx = 0
    while cap.isOpened() and len(frames) < SEQ_LEN:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        if locs:
            top, right, bottom, left = locs[0]
            top    = max(0, top - PADDING)
            bottom = min(rgb.shape[0], bottom + PADDING)
            left   = max(0, left - PADDING)
            right  = min(rgb.shape[1], right + PADDING)
            face = rgb[top:bottom, left:right]
        else:
            face = rgb
        frames.append(transform(face))
        idx += step
    cap.release()

    if not frames:
        return "no_face"

    while len(frames) < SEQ_LEN:
        frames.append(frames[-1])

    tensor = torch.stack(frames[:SEQ_LEN])
    torch.save(tensor, out_path)
    return "ok"


def run(label, video_dir):
    out_dir = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    videos  = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi"))
    print(f"\n[{label.upper()}] Processing {len(videos)} videos...")

    ok = skip = fail = 0
    for i, vpath in enumerate(videos, 1):
        out_path = out_dir / (vpath.stem + ".pt")
        status   = process_video(vpath, out_path)
        if status == "ok":    ok   += 1
        elif status == "skip": skip += 1
        else:                  fail += 1

        if i % 10 == 0 or i == len(videos):
            print(f"  {i}/{len(videos)}  saved={ok}  skipped={skip}  no_face={fail}")

    print(f"  Done — {ok} tensors saved to {out_dir}")


if __name__ == "__main__":
    run("fake", FAKE_DIR)
    run("real", REAL_DIR)
    print("\n✓ Preprocessing complete. Run benchmark.py next.")
