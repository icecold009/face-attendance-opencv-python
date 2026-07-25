import argparse
import base64
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response


BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
for import_path in (Path(__file__).resolve().parent, BASE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


from config import load_config
from attendance import AttendanceSystem
from modules.detection import detect_faces
from modules.encoding import encode_faces
from modules.identification import match_face

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


def _stack_known_encodings(
    known_encodings: List[np.ndarray],
) -> np.ndarray:
    if not known_encodings:
        return np.empty((0, 128), dtype=np.float32)
    return np.stack(
        [np.asarray(encoding, dtype=np.float32) for encoding in known_encodings],
        axis=0,
    )


def _decode_frame(frame_b64: str) -> np.ndarray | None:
    if frame_b64.startswith("data:") and "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]
    try:
        frame_data = base64.b64decode(frame_b64, validate=True)
    except (ValueError, TypeError):
        return None

    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    return frame


def _safe_person_name(name: str) -> str | None:
    normalized = name.strip()
    if not normalized or normalized in {".", ".."}:
        return None
    if "/" in normalized or "\\" in normalized:
        return None
    return normalized


def _recognize_frame(
    frame: np.ndarray,
    known_encodings: np.ndarray,
    known_labels: List[str],
    detection_model: str,
    min_confidence: float,
    frame_resize_scale: float,
    attendance_system: AttendanceSystem,
) -> Tuple[np.ndarray, List[str]]:
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=frame_resize_scale,
        fy=frame_resize_scale,
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = detect_faces(rgb_small_frame, model=detection_model)
    face_encodings = encode_faces(rgb_small_frame, face_locations)
    recognized_names: List[str] = []

    for face_encoding, (top, right, bottom, left) in zip(
        face_encodings, face_locations
    ):
        if known_encodings.size == 0:
            name, dist = "Unknown", 1.0
        else:
            name, dist = match_face(
                face_encoding,
                known_encodings,
                known_labels,
                tolerance=min_confidence,
            )

        inv_scale = 1.0 / frame_resize_scale
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
            recognized_names.append(name)
            attendance_system.mark_attendance(name)

    return frame, recognized_names


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

    def load_known_face_data() -> Tuple[np.ndarray, List[str]]:
        known_encodings, known_labels = _load_known_faces_from_folder(
            app.config["KNOWN_FACES_FOLDER"],
            app.config["DETECTION_MODEL"],
        )
        return _stack_known_encodings(known_encodings), known_labels

    def generate_frames():
        """
        Video frame generator for /video_feed.

        This opens the camera ONLY when /video_feed is requested,
        not at module import time. In CI, tests will never hit this route.
        """
        # Camera init inside generator
        video_capture = cv2.VideoCapture(0)

        # Load known faces on demand
        known_encodings_arr, known_labels = load_known_face_data()

        try:
            while True:
                success, frame = video_capture.read()
                if not success:
                    break

                frame, _recognized_names = _recognize_frame(
                    frame,
                    known_encodings_arr,
                    known_labels,
                    app.config["DETECTION_MODEL"],
                    app.config["MIN_CONFIDENCE"],
                    app.config["FRAME_RESIZE_SCALE"],
                    attendance_system,
                )

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

    @app.route("/recognize", methods=["POST"])
    def recognize():
        data = request.get_json(silent=True) or {}
        frame_b64 = data.get("frame")
        if not isinstance(frame_b64, str) or not frame_b64:
            return jsonify({"error": "No frame provided"}), 400

        frame = _decode_frame(frame_b64)
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        known_encodings, known_labels = load_known_face_data()
        annotated_frame, recognized_names = _recognize_frame(
            frame,
            known_encodings,
            known_labels,
            app.config["DETECTION_MODEL"],
            app.config["MIN_CONFIDENCE"],
            app.config["FRAME_RESIZE_SCALE"],
            attendance_system,
        )
        encoded, buffer = cv2.imencode(".jpg", annotated_frame)
        if not encoded:
            return jsonify({"error": "Failed to encode result frame"}), 500

        return jsonify(
            {
                "success": True,
                "annotated_frame": base64.b64encode(buffer).decode("ascii"),
                "recognized_names": recognized_names,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    @app.route("/enroll", methods=["POST"])
    def enroll():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        frame_b64 = data.get("frame")
        if not isinstance(name, str) or _safe_person_name(name) is None:
            return jsonify({"error": "A valid name is required"}), 400
        if not isinstance(frame_b64, str) or not frame_b64:
            return jsonify({"error": "No frame provided"}), 400

        frame = _decode_frame(frame_b64)
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = detect_faces(
            rgb_frame,
            model=app.config["DETECTION_MODEL"],
        )
        if not encode_faces(rgb_frame, face_locations):
            return jsonify({"error": "No face detected"}), 400

        person_name = _safe_person_name(name)
        person_dir = app.config["KNOWN_FACES_FOLDER"] / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        image_path = person_dir / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        )
        if not cv2.imwrite(str(image_path), frame):
            return jsonify({"error": "Failed to save enrollment image"}), 500

        return jsonify(
            {
                "success": True,
                "message": f"Successfully enrolled {person_name}",
            }
        )

    @app.route("/attendance", methods=["GET"])
    def get_attendance():
        summary = attendance_system.get_attendance_summary()
        attendance = [] if summary is None else summary.to_dict(orient="records")
        return jsonify(
            {
                "success": True,
                "attendance": attendance,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "count": len(attendance),
            }
        )

    @app.route("/enrolled-persons", methods=["GET"])
    def get_enrolled_persons():
        known_faces_folder = app.config["KNOWN_FACES_FOLDER"]
        persons = sorted(
            directory.name
            for directory in known_faces_folder.iterdir()
            if directory.is_dir()
        ) if known_faces_folder.exists() else []
        return jsonify({"success": True, "persons": persons, "count": len(persons)})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

    return app


# Optional: run directly for local dev
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Attendance App")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind (default: 5000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    app_instance = create_app()
    app_instance.run(debug=args.debug, host=args.host, port=args.port)
