from flask import Flask, render_template, request
import os
import cv2
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Project folders
PROJECT_FOLDERS = [

    'uploads',
    'extracted_frames',
    'cropped_faces',
    'reports',
    'temp'

]

# Create folders automatically
for folder in PROJECT_FOLDERS:

    os.makedirs(folder, exist_ok=True)

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Maximum file size = 500 MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Allowed formats
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

# Create uploads folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Verify extension
def allowed_file(filename):

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home page
@app.route('/')
def home():

    return render_template('index.html')


# Upload route
@app.route('/upload', methods=['POST'])
def upload_video():

    # Check request
    if 'video' not in request.files:

        return render_template(
            'index.html',
            error='No file uploaded'
        )

    file = request.files['video']

    # Empty filename
    if file.filename == '':

        return render_template(
            'index.html',
            error='No selected file'
        )

    # Invalid extension
    if not allowed_file(file.filename):

        return render_template(
            'index.html',
            error='Invalid format. Only MP4 and AVI allowed.'
        )

    # Unique filename
    filename = (
        str(uuid.uuid4())
        + '_'
        + secure_filename(file.filename)
    )

    # Full save path
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    # Save video
    file.save(filepath)

    # =========================
    # COMMUNICATION WITH OTHER MODULES
    # =========================

    video_path = filepath

    frames_output_folder = os.path.join(
        'extracted_frames',
        filename.split('.')[0]
    )

    os.makedirs(
        frames_output_folder,
        exist_ok=True
    )

    # Open video
    video = cv2.VideoCapture(filepath)

    # Corrupted video check
    if not video.isOpened():

        return render_template(
            'index.html',
            error='Corrupted or unreadable video.'
        )

    # FPS
    fps = video.get(cv2.CAP_PROP_FPS)

    # Total frames
    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Duration
    duration = frame_count / fps

    # Resolution
    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # File size
    file_size = round(
        os.path.getsize(filepath)
        / (1024 * 1024),
        2
    )

    video.release()

    print("UPLOAD SUCCESSFUL")
    print(filepath)

    return render_template(

        'success.html',

        filename=filename,

        fps=round(fps, 2),

        duration=round(duration, 2),

        width=width,

        height=height,

        frame_count=frame_count,

        file_size=file_size,

        video_path=video_path,

        frames_output_folder=frames_output_folder
    )
# Large file error
@app.errorhandler(413)
def too_large(e):

    return render_template(
        'index.html',
        error='File too large. Maximum size is 500 MB.'
    )


if __name__ == '__main__':

    app.run(debug=True)