"""
Step 2 & 3 & 4: Unzip, detect structure, and organize dataset.

Usage:
    python src/prepare_dataset.py --zip_path /path/to/dataset.zip

What it does:
    1. Extracts the zip file into a temporary directory.
    2. Scans the extracted structure.
    3. Detects whether images are already in train/val/test with real/fake subfolders.
    4. If not organized, splits them into the wanted format:
         dataset/
           train/real/
           train/fake/
           val/real/
           val/fake/
           test/real/
           test/fake/
"""

import argparse
import zipfile
import shutil
import random
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
)


def extract_zip(zip_path, extract_dir):
    print(f"[1/4] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[+] Extracted to {extract_dir}")


def detect_structure(extract_dir):
    print("[2/4] Detecting dataset structure ...")
    all_items = list(extract_dir.rglob("*"))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = [p for p in all_items if p.suffix.lower() in image_extensions]

    if not image_paths:
        raise RuntimeError("No images found in the zip file.")

    print(f"    Found {len(image_paths)} images total.")

    # Check if already in real/fake subfolders
    subdirs = set()
    for p in image_paths:
        rel = p.relative_to(extract_dir)
        if len(rel.parts) >= 2:
            subdirs.add(rel.parts[0])
            subdirs.add(rel.parts[-2].lower())

    has_real_fake = "real" in subdirs or "fake" in subdirs
    print(f"    Detected real/fake folders: {has_real_fake}")

    return image_paths, has_real_fake


def organize_dataset(image_paths, has_real_fake, extract_dir):
    print("[3/4] Organizing dataset into train/val/test with real/fake ...")

    # Remove old organized dataset if it exists
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if d.exists():
            shutil.rmtree(d)
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (d / "real").mkdir(parents=True, exist_ok=True)
        (d / "fake").mkdir(parents=True, exist_ok=True)

    if has_real_fake:
        real_images = [p for p in image_paths if "real" in p.parts]
        fake_images = [p for p in image_paths if "fake" in p.parts]
    else:
        # If no real/fake labels, ask user to label or raise error
        raise RuntimeError(
            "Dataset does not contain 'real'/'fake' subfolders. "
            "Please ensure your zip contains images organized as:\n"
            "  real/  and  fake/  folders (possibly inside train/val/test)."
        )

    print(f"    Real images: {len(real_images)}")
    print(f"    Fake images: {len(fake_images)}")

    random.seed(RANDOM_SEED)

    def split_and_copy(images, label):
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

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

    print("[4/4] Dataset organization complete!")
    print(f"    Train: real={len(list(TRAIN_DIR.glob('real/*')))}  fake={len(list(TRAIN_DIR.glob('fake/*')))}")
    print(f"    Val:   real={len(list(VAL_DIR.glob('real/*')))}  fake={len(list(VAL_DIR.glob('fake/*')))}")
    print(f"    Test:  real={len(list(TEST_DIR.glob('real/*')))}  fake={len(list(TEST_DIR.glob('fake/*')))}")


def main():
    parser = argparse.ArgumentParser(description="Prepare deepfake dataset")
    parser.add_argument("--zip_path", type=str, default=None,
                        help="Path to the dataset zip file")
    args = parser.parse_args()

    zip_path = Path(args.zip_path) if args.zip_path else DATASET_ZIP
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    extract_dir = DATASET_EXTRACT_DIR
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    extract_zip(zip_path, extract_dir)
    image_paths, has_real_fake = detect_structure(extract_dir)
    organize_dataset(image_paths, has_real_fake, extract_dir)

    # Clean up extraction dir
    shutil.rmtree(extract_dir)
    print("[+] Temporary extraction removed.")


if __name__ == "__main__":
    main()
