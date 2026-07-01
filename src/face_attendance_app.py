import os
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, Response

from config import load_config
from modules.detection import detect_faces
from modules.encoding import encode_faces
from modules.identification import match_face


# -------------------------------------------------
# Configuration and global state
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
cfg = load_config()

DETECTION_MODEL = cfg.get("detection_model", "hog")
MIN_CONFIDENCE = cfg.get("min_confidence", 0.6)
FRAME_RESIZE_SCALE = cfg.get("frame_resize_scale", 0.25)

ATTENDANCE_CSV_PATH = BASE_DIR / cfg.get(
    "attendance_csv_path", "data/Attendance/attendance.csv"
)

# TODO: adjust these paths / loading logic to match your repo
KNOWN_ENCODINGS_PATH = BASE_DIR / "ImagesAttendance"  # or wherever you store known faces


# -------------------------------------------------
# Helpers for loading known faces and attendance
# -------------------------------------------------

def load_known_faces_from_folder(folder_path: Path):
    """
    Load known face encodings and labels from a folder structure.

    Expected structure:
        folder_path/
            person1/ img1.jpg, img2.jpg, ...
            person2/ img1.jpg, ...

    Returns:
        known_encodings: List[np.ndarray]
        known_labels: List[str]
    """
    import face_recognition  # still needed for initial encoding

    known_encodings = []
    known_labels = []

    for person_name in os.listdir(folder_path):
        person_dir = folder_path / person_name
        if not person_dir.is_dir():
            continue
        for fname in os.listdir(person_dir):
            img_path = person_dir / fname
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = detect_faces(rgb_img, model=DETECTION_MODEL)
            encs = encode_faces(rgb_img, boxes)
            if not encs:
                continue
            known_encodings.append(encs[0])
            known_labels.append(person_name)

    return known_encodings, known_labels


def ensure_attendance_csv_exists():
    """
    Create the attendance CSV file with header if it does not exist.
    """
    ATTENDANCE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ATTENDANCE_CSV_PATH.exists():
        with ATTENDANCE_CSV_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "timestamp", "confidence"])


def mark_attendance(name: str, confidence: float):
    """
    Append a new attendance record for a recognized person.
    """
    ensure_attendance_csv_exists()
    timestamp = datetime.now().isoformat(timespec="seconds")
    with ATTENDANCE_CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, timestamp, f"{confidence:.4f}"])


# -------------------------------------------------
# Video capture + generator
# -------------------------------------------------

video_capture = cv2.VideoCapture(0)

known_encodings, known_labels = load_known_faces_from_folder(KNOWN_ENCODINGS_PATH)


def generate_frames():
    """
    Video frame generator that performs detection + encoding + identification
    on each frame and yields JPEG-encoded frames for Flask streaming.
    """
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # optional resize for speed
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=FRAME_RESIZE_SCALE,
            fy=FRAME_RESIZE_SCALE,
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = detect_faces(rgb_small_frame, model=DETECTION_MODEL)
        face_encodings = encode_faces(rgb_small_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(
            face_encodings, face_locations
        ):
            name, dist = match_face(
                face_encoding,
                known_encodings,
                known_labels,
                tolerance=MIN_CONFIDENCE,
            )

            # scale back up face locations since the frame was resized
            top = int(top / FRAME_RESIZE_SCALE)
            right = int(right / FRAME_RESIZE_SCALE)
            bottom = int(bottom / FRAME_RESIZE_SCALE)
            left = int(left / FRAME_RESIZE_SCALE)

            # draw bounding box and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({dist:.2f})"
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if name != "Unknown":
                mark_attendance(name, dist)

        # encode frame for streaming
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# -------------------------------------------------
# Flask app
# -------------------------------------------------

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()