import cv2
import os
import time
from datetime import datetime

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)


def extract_frames(video_path, output_folder):

    # Current date
    current_date = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # Video name
    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # User name
    user_name = "user_1"

    # Organized output folder
    output_folder = os.path.join(
        "extracted_frames",
        user_name,
        current_date,
        video_name
    )

    # Create folders
    os.makedirs(output_folder, exist_ok=True)

    # Open video
    video = cv2.VideoCapture(video_path)

    # Counters
    frame_count = 0
    saved_count = 0
    error_count = 0

    # Check video
    if not video.isOpened():

        print("Error opening video")

        error_count += 1

        return

    # Start timer
    start_time = time.time()

    while True:

        # Read frame
        success, frame = video.read()

        # End of video
        if not success:
            break

        # Save 1 frame every 10 frames
        if frame_count % 10 == 0:

            # Convert to grayscale
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Blur detection
            variance = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()

            print("Variance:", variance)

            # Ignore blurry frames
            if variance < 1:

                frame_count += 1
                continue

            # Brightness detection
            brightness = gray.mean()

            print("Brightness:", brightness)

            # Ignore dark frames
            if brightness < 30:

                frame_count += 1
                continue

            # Face detection
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            print("Faces detected:", len(faces))

            # Ignore frames without faces
            if len(faces) == 0:

                frame_count += 1
                continue

            # Draw rectangle around faces
            for (x, y, w, h) in faces:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            # Convert BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize frame
            frame = cv2.resize(
                frame,
                (224, 224)
            )

            print("Frame resized:", frame.shape)

            # Save path
            frame_filename = os.path.join(
                output_folder,
                f"frame_{saved_count}.jpg"
            )

            # Save image
            cv2.imwrite(
                frame_filename,
                frame
            )

            print(f"Saved: {frame_filename}")

            saved_count += 1

        frame_count += 1

    video.release()

    # End timer
    end_time = time.time()

    execution_time = round(
        end_time - start_time,
        2
    )

    # Logs
    log_message = f"""
Video: {video_path}
Frames extracted: {saved_count}
Execution time: {execution_time} sec
Errors: {error_count}

-------------------------
"""

    with open("logs.txt", "a") as log_file:

        log_file.write(log_message)

    print(f"\n{saved_count} frames extracted successfully.")


# TEST
extract_frames(
    "uploads/test.mp4",
    "extracted_frames/test_video"
)