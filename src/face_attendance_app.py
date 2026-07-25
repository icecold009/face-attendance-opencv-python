import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from flask import Flask, render_template, Response

from config import load_config
from attendance import AttendanceSystem
from modules.detection import detect_faces
from modules.encoding import encode_faces
from modules.identification import match_face


BASE_DIR = Path(__file__).resolve().parents[1]  # repo root


def _load_known_faces_from_folder(
    folder_path: Path,
    detection_model: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load known face encodings and labels from a folder structure.

    Expected structure:
        folder_path/
            person1/ img1.jpg, img2.jpg, ...
            person2/ img1.jpg, ...

    This runs only when the app actually starts, not at import time.
    """
    known_encodings: List[np.ndarray] = []
    known_labels: List[str] = []

    if not folder_path.exists():
        # In CI or fresh environments, just return empty lists
        return known_encodings, known_labels

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
            boxes = detect_faces(rgb_img, model=detection_model)
            encs = encode_faces(rgb_img, boxes)
            if not encs:
                continue
            known_encodings.append(encs[0])
            known_labels.append(person_name)

    return known_encodings, known_labels


def create_app() -> Flask:
    """
    Application factory.

    This function is safe to call in CI tests:
    - It does not open the camera.
    - It does not require data folders to exist.
    """
    cfg = load_config()

    detection_model = cfg.get("detection_model", "hog")
    min_confidence = cfg.get("min_confidence", 0.6)
    frame_resize_scale = cfg.get("frame_resize_scale", 0.25)

    attendance_csv_path = BASE_DIR / cfg.get(
        "attendance_csv_path", "data/Attendance/attendance.csv"
    )
    attendance_path = attendance_csv_path.parent
    attendance_system = AttendanceSystem(str(attendance_path))

    known_faces_folder = BASE_DIR / "ImagesAttendance"

    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

    # Store config in app context so routes can use it
    app.config["DETECTION_MODEL"] = detection_model
    app.config["MIN_CONFIDENCE"] = float(min_confidence)
    app.config["FRAME_RESIZE_SCALE"] = float(frame_resize_scale)
    app.config["ATTENDANCE_PATH"] = attendance_path
    app.config["ATTENDANCE_SYSTEM"] = attendance_system
    app.config["KNOWN_FACES_FOLDER"] = known_faces_folder

    @app.route("/")
    def index():
        return render_template("index.html")

    def generate_frames():
        """
        Video frame generator for /video_feed.

        This opens the camera ONLY when /video_feed is requested,
        not at module import time. In CI, tests will never hit this route.
        """
        # Camera init inside generator
        video_capture = cv2.VideoCapture(0)

        # Load known faces on demand
        known_encodings, known_labels = _load_known_faces_from_folder(
            app.config["KNOWN_FACES_FOLDER"],
            app.config["DETECTION_MODEL"],
        )
        if len(known_encodings) > 0:
            # Ensure they are numpy arrays and stack along axis 0
            enc_list = [np.asarray(e, dtype=np.float32) for e in known_encodings]
            known_encodings_arr = np.stack(enc_list, axis=0)
        else:
            known_encodings_arr = np.empty((0, 128), dtype=np.float32)

        try:
            while True:
                success, frame = video_capture.read()
                if not success:
                    break

                small_frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=app.config["FRAME_RESIZE_SCALE"],
                    fy=app.config["FRAME_RESIZE_SCALE"],
                )
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = detect_faces(
                    rgb_small_frame,
                    model=app.config["DETECTION_MODEL"],
                )
                face_encodings = encode_faces(rgb_small_frame, face_locations)

                for face_encoding, (top, right, bottom, left) in zip(
                    face_encodings, face_locations
                ):
                    if known_encodings_arr.size == 0:
                        name, dist = "Unknown", 1.0
                    else:
                        name, dist = match_face(
                            face_encoding,
                            known_encodings_arr,
                            known_labels,
                            tolerance=app.config["MIN_CONFIDENCE"],
                        )

                    inv_scale = 1.0 / app.config["FRAME_RESIZE_SCALE"]
                    top = int(top * inv_scale)
                    right = int(right * inv_scale)
                    bottom = int(bottom * inv_scale)
                    left = int(left * inv_scale)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    label = f"{name} ({dist:.2f})"
                    cv2.rectangle(
                        frame,
                        (left, bottom - 20),
                        (right, bottom),
                        color,
                        cv2.FILLED,
                    )
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
                        attendance_system.mark_attendance(name)

                ret, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


# Optional: run directly for local dev
if __name__ == "__main__":
    app_instance = create_app()
    app_instance.run(host="0.0.0.0", port=5000, debug=True)
