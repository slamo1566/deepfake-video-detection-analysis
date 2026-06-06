import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def detect_faces(image_path):
    """
    Détecte les visages dans une image.
    Retourne liste de dicts avec bbox et landmarks (yeux).
    """
    img = cv2.imread(image_path)
    if img is None:
        return [], None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = []
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as detector:
        results = detector.process(rgb)

    if not results.detections:
        return [], img

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box

        # Coordonnées absolues
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)

        # Points des yeux (pour alignement)
        kp = det.location_data.relative_keypoints
        left_eye  = (int(kp[0].x * w), int(kp[0].y * h))
        right_eye = (int(kp[1].x * w), int(kp[1].y * h))

        if bw > 10 and bh > 10:
            faces.append({
                "bbox": (x, y, bw, bh),
                "left_eye": left_eye,
                "right_eye": right_eye
            })

    return faces, img


if __name__ == "__main__":
    import os
    TEST_DIR = "extracted_frames"
    found = False
    for subfolder in os.listdir(TEST_DIR):
        sub_path = os.path.join(TEST_DIR, subfolder)
        if not os.path.isdir(sub_path):
            continue
        for fname in os.listdir(sub_path):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_img = os.path.join(sub_path, fname)
                faces, _ = detect_faces(test_img)
                print(f"Test : {fname} → {len(faces)} visage(s) détecté(s)")
                found = True
                break
        if found:
            break
    if not found:
        print("Aucune image trouvée dans extracted_frames/")