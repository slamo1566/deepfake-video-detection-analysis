import cv2
import numpy as np
import os
import shutil
from face_detector import detect_faces

# ── Configuration ──────────────────────────────────────────
INPUT_DIR  = "extracted_frames"
OUTPUT_DIR = "processed_faces"
IMG_SIZE   = (224, 224)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# ───────────────────────────────────────────────────────────


def align_face(img, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def clean_image(face):
    face = cv2.fastNlMeansDenoisingColored(face, None, 5, 5, 7, 21)
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def is_usable(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20:
        return False
    brightness = np.mean(gray)
    if brightness < 20 or brightness > 235:
        return False
    return True


def preprocess_face(img, face_info):
    x, y, w, h = face_info["bbox"]
    aligned = align_face(img, face_info["left_eye"], face_info["right_eye"])
    face = aligned[y:y+h, x:x+w]
    if face.size == 0:
        return None
    if not is_usable(face):
        return None
    face = clean_image(face)
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face = face.astype(np.float32) / 255.0
    return face


def get_all_image_files(root_dir):
    """
    Parcourt TOUS les sous-dossiers récursivement.
    Retourne liste de (chemin_complet, label) où label = 'fake' ou 'real'.
    """
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(dirpath, fname)
                # Si "fake" apparaît n'importe où dans le chemin → fake
                label = "fake" if "fake" in dirpath.lower() else "real"
                image_files.append((full_path, fname, label))
    return image_files


def collect_all_faces(image_files):
    real_faces = []
    fake_faces = []

    for (img_path, fname, label) in image_files:
        faces, img = detect_faces(img_path)
        if not faces or img is None:
            continue
        best = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        processed = preprocess_face(img, best)
        if processed is not None:
            if label == "fake":
                fake_faces.append((fname, processed))
            else:
                real_faces.append((fname, processed))

    return real_faces, fake_faces


def split_and_save(faces_list, label):
    items = faces_list.copy()
    np.random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": items[:n_train],
        "val":   items[n_train:n_train + n_val],
        "test":  items[n_train + n_val:]
    }

    for split_name, split_items in splits.items():
        out_dir = os.path.join(OUTPUT_DIR, label, split_name)
        os.makedirs(out_dir, exist_ok=True)
        for (fname, face) in split_items:
            cv2.imwrite(os.path.join(out_dir, fname), (face * 255).astype(np.uint8))

    return {k: len(v) for k, v in splits.items()}


def run_preprocessing():
    print("=" * 50)
    print("  MEMBRE 3 — Preprocessing des visages")
    print("=" * 50)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("\n[1/2] Scan de tous les dossiers...")
    all_files = get_all_image_files(INPUT_DIR)
    real_files = [(p, f, l) for (p, f, l) in all_files if l == "real"]
    fake_files = [(p, f, l) for (p, f, l) in all_files if l == "fake"]
    print(f"      → {len(real_files)} frames real trouvées")
    print(f"      → {len(fake_files)} frames fake trouvées")

    print("\n[2/2] Détection et traitement des visages...")
    real_faces, fake_faces = collect_all_faces(all_files)
    print(f"      → {len(real_faces)} visages real exploitables")
    print(f"      → {len(fake_faces)} visages fake exploitables")

    print("\nSauvegarde du dataset...")
    stats_real = split_and_save(real_faces, "real")
    stats_fake = split_and_save(fake_faces, "fake")

    print(f"\n✅ Dataset prêt dans processed_faces/")
    print(f"\n   REAL → train: {stats_real['train']} | val: {stats_real['val']} | test: {stats_real['test']}")
    print(f"   FAKE → train: {stats_fake['train']} | val: {stats_fake['val']} | test: {stats_fake['test']}")
    print(f"\n   Structure finale :")
    print(f"   processed_faces/")
    print(f"   ├── real/  ├── train/  ├── val/  └── test/")
    print(f"   └── fake/  ├── train/  ├── val/  └── test/")


if __name__ == "__main__":
    run_preprocessing()
    