import cv2
import numpy as np
import os
import shutil
from face_detector import detect_faces

# ── Configuration ──────────────────────────────────────────
INPUT_DIR  = "extracted_frames"
OUTPUT_DIR = "processed_faces"
IMG_SIZE   = (224, 224)

# Répartition dataset  (total = 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
# ───────────────────────────────────────────────────────────


def align_face(img, left_eye, right_eye):
    """Aligne le visage pour que les yeux soient horizontaux."""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR)
    return aligned


def clean_image(face):
    """Réduction bruit + correction luminosité + amélioration contraste."""
    # Réduction du bruit
    face = cv2.fastNlMeansDenoisingColored(face, None, 5, 5, 7, 21)

    # Correction luminosité / contraste via CLAHE sur canal L
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return face


def is_usable(face):
    """Vérifie si l'image est exploitable (pas trop sombre/floue)."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Vérif flou (variance du Laplacien)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 20:
        return False

    # Vérif luminosité moyenne
    brightness = np.mean(gray)
    if brightness < 20 or brightness > 235:
        return False

    return True


def preprocess_face(img, face_info):
    """Pipeline complet : align → crop → clean → resize → normalize."""
    x, y, w, h = face_info["bbox"]

    # 1. Alignement
    aligned = align_face(img, face_info["left_eye"], face_info["right_eye"])

    # 2. Crop ROI
    face = aligned[y:y+h, x:x+w]
    if face.size == 0:
        return None

    # 3. Nettoyage
    if not is_usable(face):
        return None
    face = clean_image(face)

    # 4. Resize 224×224
    face = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # 5. Normalisation [0, 1]
    face = face.astype(np.float32) / 255.0

    return face


def collect_all_faces():
    """Parcourt extracted_frames/ et retourne la liste de toutes les faces traitées."""
    results = []  # liste de (subfolder_name, filename, face_array)

    for subfolder in sorted(os.listdir(INPUT_DIR)):
        sub_path = os.path.join(INPUT_DIR, subfolder)
        if not os.path.isdir(sub_path):
            continue

        frames = sorted([
            f for f in os.listdir(sub_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        for fname in frames:
            img_path = os.path.join(sub_path, fname)
            faces, img = detect_faces(img_path)

            if not faces or img is None:
                continue

            # Prendre le visage le plus grand
            best = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
            processed = preprocess_face(img, best)

            if processed is not None:
                results.append((subfolder, fname, processed))

    return results


def split_and_save(faces_list, label):
    """
    Divise la liste en train/val/test et sauvegarde dans :
    processed_faces/<label>/train|val|test/
    """
    np.random.shuffle(faces_list)
    n = len(faces_list)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": faces_list[:n_train],
        "val":   faces_list[n_train:n_train + n_val],
        "test":  faces_list[n_train + n_val:]
    }

    for split_name, items in splits.items():
        out_dir = os.path.join(OUTPUT_DIR, label, split_name)
        os.makedirs(out_dir, exist_ok=True)
        for (_, fname, face) in items:
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, (face * 255).astype(np.uint8))

    return {k: len(v) for k, v in splits.items()}


def run_preprocessing():
    print("=" * 50)
    print("  MEMBRE 3 — Preprocessing des visages")
    print("=" * 50)

    # Nettoyage ancien output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("\n[1/2] Détection et traitement des visages...")
    all_faces = collect_all_faces()
    print(f"      → {len(all_faces)} visages exploitables trouvés")

    if len(all_faces) == 0:
        print("\n⚠️  Aucun visage trouvé. Vérifie le dossier extracted_frames/")
        return

    # ── Classement real / fake ──────────────────────────────
    # Les sous-dossiers contenant "fake" dans leur nom → fake
    # Les autres → real
    real_faces = [(s, f, img) for (s, f, img) in all_faces
                  if "fake" not in s.lower()]
    fake_faces = [(s, f, img) for (s, f, img) in all_faces
                  if "fake" in s.lower()]

    # Si aucune distinction possible, tout mettre en real pour l'instant
    if not fake_faces:
        print("\n⚠️  Aucun dossier 'fake' détecté.")
        print("    Toutes les faces sont classées dans 'real/'.")
        print("    Renomme tes sous-dossiers avec 'fake' dans le nom pour la classification.")
        real_faces = all_faces

    print("\n[2/2] Sauvegarde du dataset final...")

    stats_real = split_and_save(real_faces, "real")
    stats_fake = split_and_save(fake_faces, "fake") if fake_faces else {}

    print("\n✅ Dataset prêt dans processed_faces/")
    print(f"\n   REAL  → train: {stats_real.get('train',0)} | "
          f"val: {stats_real.get('val',0)} | "
          f"test: {stats_real.get('test',0)}")
    if stats_fake:
        print(f"   FAKE  → train: {stats_fake.get('train',0)} | "
              f"val: {stats_fake.get('val',0)} | "
              f"test: {stats_fake.get('test',0)}")

    print(f"\n   Structure finale :")
    print(f"   processed_faces/")
    print(f"   ├── real/")
    print(f"   │   ├── train/")
    print(f"   │   ├── val/")
    print(f"   │   └── test/")
    print(f"   └── fake/")
    print(f"       ├── train/")
    print(f"       ├── val/")
    print(f"       └── test/")


if __name__ == "__main__":
    run_preprocessing()