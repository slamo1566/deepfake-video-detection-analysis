"""
Unzip, detect structure, and organize the DFD video dataset.

Workflow:
  1. Extract zip file
  2. Detect whether it contains images or videos (DFD dataset is videos)
  3. Separate into dataset/videos/real/  and  dataset/videos/fake/
  4. Optionally extract frames from videos (so we can train without waiting for members 2 & 3)
  5. Optionally detect & crop faces from frames using OpenCV

Usage:
    # Step 1: just organize videos
    python src/prepare_dataset.py --zip_path /path/to/archive.zip

    # Step 2: also extract frames (every 10th frame) for training
    python src/prepare_dataset.py --zip_path /path/to/archive.zip --extract-frames

    # Step 3: extract frames + crop faces
    python src/prepare_dataset.py --zip_path /path/to/archive.zip --extract-frames --detect-faces
"""

import argparse
import zipfile
import shutil
import random
import cv2
from pathlib import Path
from src.config import (
    DATASET_DIR,
    DATASET_EXTRACT_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    IMG_SIZE,
    VIDEOS_DIR,
    VIDEOS_REAL_DIR,
    VIDEOS_FAKE_DIR,
    FRAME_EXTRACT_RATE,
)


def extract_zip(zip_path, extract_dir):
    print(f"[1] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"    Extracted to {extract_dir}")


def detect_structure(extract_dir):
    print("[2] Detecting dataset structure ...")
    all_items = list(extract_dir.rglob("*"))
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    images = [p for p in all_items if p.suffix.lower() in image_ext]
    videos = [p for p in all_items if p.suffix.lower() in video_ext]

    print(f"    Found {len(images)} images, {len(videos)} videos")

    # Check for real/fake keywords in folder names
    folder_names = set()
    for p in all_items:
        for part in p.parts:
            folder_names.add(part.lower())

    has_real_fake = "real" in folder_names or "fake" in folder_names
    if not has_real_fake:
        has_real_fake = "original" in folder_names or "manipulated" in folder_names
    print(f"    Detected real/fake folders: {has_real_fake}")

    return images, videos, has_real_fake, folder_names


def organize_videos(videos, folder_names, extract_dir):
    print("[3] Organizing videos into dataset/videos/real/ and dataset/videos/fake/ ...")

    if VIDEOS_DIR.exists():
        shutil.rmtree(VIDEOS_DIR)
    VIDEOS_REAL_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_FAKE_DIR.mkdir(parents=True, exist_ok=True)

    # Detect which folders map to real vs fake
    def classify_by_folder(video):
        parts_lower = [p.lower() for p in video.parts]
        if any("original" in p for p in parts_lower):
            return "real"
        if any("manipulated" in p for p in parts_lower):
            return "fake"
        if any("real" in p for p in parts_lower):
            return "real"
        if any("fake" in p for p in parts_lower):
            return "fake"
        return None

    real_videos = []
    fake_videos = []
    unlabeled = []

    for v in videos:
        label = classify_by_folder(v)
        if label == "real":
            real_videos.append(v)
        elif label == "fake":
            fake_videos.append(v)
        else:
            unlabeled.append(v)

    print(f"    Real videos: {len(real_videos)}")
    print(f"    Fake videos: {len(fake_videos)}")
    if unlabeled:
        print(f"    WARNING: {len(unlabeled)} videos could not be classified (no 'original'/'manipulated' in path)")

    def copy_videos(src_list, dest_dir):
        for src in src_list:
            dst = dest_dir / src.name
            if dst.exists():
                # Avoid name collision by prefixing parent folder
                parent_folder = src.parent.name
                dst = dest_dir / f"{parent_folder}__{src.name}"
            shutil.copy2(str(src), str(dst))

    copy_videos(real_videos, VIDEOS_REAL_DIR)
    copy_videos(fake_videos, VIDEOS_FAKE_DIR)

    print(f"    Videos organized: real={len(list(VIDEOS_REAL_DIR.iterdir()))}  fake={len(list(VIDEOS_FAKE_DIR.iterdir()))}")


def extract_frames_from_videos(src_dir, dest_dir, label, rate=FRAME_EXTRACT_RATE):
    """Extract frames from all videos in src_dir, every `rate` frames."""
    video_files = list(src_dir.glob("*"))
    if not video_files:
        print(f"    No videos found in {src_dir}")
        return

    (dest_dir / label).mkdir(parents=True, exist_ok=True)
    total_frames = 0

    for vpath in video_files:
        cap = cv2.VideoCapture(str(vpath))
        frame_count = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % rate == 0:
                out_name = f"{vpath.stem}_frame{frame_count:06d}.jpg"
                out_path = dest_dir / label / out_name
                cv2.imwrite(str(out_path), frame)
                saved += 1
            frame_count += 1
        cap.release()
        total_frames += saved
        print(f"      {vpath.name}: {saved} frames extracted")

    print(f"    Total frames saved to {dest_dir / label}/: {total_frames}")
    return total_frames


def detect_and_crop_faces(src_dir, dest_dir, label):
    """Run OpenCV Haar Cascade face detection and crop faces."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    image_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    if not image_files:
        print(f"    No images found in {src_dir}")
        return

    (dest_dir / label).mkdir(parents=True, exist_ok=True)
    total_faces = 0
    no_face = 0

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            no_face += 1
            continue

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y : y + h, x : x + w]
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            out_name = f"{img_path.stem}_face{i}.jpg"
            out_path = dest_dir / label / out_name
            cv2.imwrite(str(out_path), face_resized)
            total_faces += 1

    print(f"    Faces cropped: {total_faces}  (no face detected: {no_face})")


def split_faces_into_train_val_test(faces_dir):
    """Copy face images from faces_dir/real and faces_dir/fake into train/val/test."""
    print("[4] Splitting face images into train/val/test ...")

    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
        (d / "real").mkdir(parents=True, exist_ok=True)
        (d / "fake").mkdir(parents=True, exist_ok=True)

    real_images = list((faces_dir / "real").glob("*"))
    fake_images = list((faces_dir / "fake").glob("*"))

    random.seed(RANDOM_SEED)

    def split_and_copy(images, label):
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        for img_list, dest_dir in [
            (train_imgs, TRAIN_DIR / label),
            (val_imgs, VAL_DIR / label),
            (test_imgs, TEST_DIR / label),
        ]:
            for src in img_list:
                dst = dest_dir / src.name
                shutil.copy2(str(src), str(dst))
        print(f"    {label}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    split_and_copy(real_images, "real")
    split_and_copy(fake_images, "fake")

    print(f"    Train: real={len(list((TRAIN_DIR/'real').iterdir()))}  fake={len(list((TRAIN_DIR/'fake').iterdir()))}")
    print(f"    Val:   real={len(list((VAL_DIR/'real').iterdir()))}  fake={len(list((VAL_DIR/'fake').iterdir()))}")
    print(f"    Test:  real={len(list((TEST_DIR/'real').iterdir()))}  fake={len(list((TEST_DIR/'fake').iterdir()))}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DFD video dataset for training")
    parser.add_argument("--zip_path", type=str, default=None, help="Path to dataset zip file")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from videos")
    parser.add_argument("--detect-faces", action="store_true", help="Detect & crop faces from frames")
    parser.add_argument("--frame-rate", type=int, default=FRAME_EXTRACT_RATE, help="Extract every Nth frame")
    args = parser.parse_args()

    zip_path = Path(args.zip_path) if args.zip_path else DATASET_ZIP
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    # Step A: extract zip
    extract_dir = DATASET_EXTRACT_DIR
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    extract_zip(zip_path, extract_dir)

    # Step B: detect structure
    images, videos, has_real_fake, folder_names = detect_structure(extract_dir)

    # Step C: organize (videos or images)
    if videos:
        organize_videos(videos, folder_names, extract_dir)
        print("\n    NOTE: dataset contains VIDEOS. To train, you need face images.")
        print("    Option A: run with --extract-frames --detect-faces to generate them now.")
        print("    Option B: wait for members 2 & 3 to provide processed face images.\n")
    else:
        raise RuntimeError("No video files (.mp4, .avi, etc.) found in the zip.")

    # Step D: optional frame extraction
    if args.extract_frames or args.detect_faces:
        frames_dir = DATASET_DIR / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)

        print("\n[Optional] Extracting frames from videos ...")
        extract_frames_from_videos(VIDEOS_REAL_DIR, frames_dir, "real", args.frame_rate)
        extract_frames_from_videos(VIDEOS_FAKE_DIR, frames_dir, "fake", args.frame_rate)

    # Step E: optional face detection
    if args.detect_faces:
        faces_dir = DATASET_DIR / "faces"
        if faces_dir.exists():
            shutil.rmtree(faces_dir)

        print("\n[Optional] Detecting and cropping faces ...")
        detect_and_crop_faces(frames_dir / "real", faces_dir, "real")
        detect_and_crop_faces(frames_dir / "fake", faces_dir, "fake")

    # Step F: split into train/val/test if faces were extracted
    if args.detect_faces:
        split_faces_into_train_val_test(faces_dir)
        print("\n[+] Dataset ready for training!")
    elif args.extract_frames:
        print(f"\n[+] Frames extracted to {DATASET_DIR / 'frames'}/")
        print("    Run with --detect-faces next to crop faces, or use member 3's output.")

    # Cleanup
    shutil.rmtree(extract_dir)
    print("[+] Temporary extraction removed.")


if __name__ == "__main__":
    main()
