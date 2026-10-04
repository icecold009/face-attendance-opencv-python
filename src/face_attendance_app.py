import argparse
import base64
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response


BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
for import_path in (Path(__file__).resolve().parent, BASE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


from config import load_config
from attendance import AttendanceSystem
from attendance_adapters import FaceRecognitionAdapter, OpenCVImageStore, SystemClock
from attendance_services.enrollment import (
    EnrollmentService,
    EnrollmentStatus,
    TimestampEnrollmentIdGenerator,
    normalize_person_name,
)
from attendance_services.recognition import RecognitionSession, RecognitionService
from modules.detection import detect_faces
from modules.encoding import encode_faces
from modules.identification import match_face

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
    """Compatibility wrapper for the service's pure name validation rule."""
    return normalize_person_name(name)


def _recognize_frame(
    frame: np.ndarray,
    recognition_session: RecognitionSession,
    frame_resize_scale: float,
) -> Tuple[np.ndarray, List[str]]:
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=frame_resize_scale,
        fy=frame_resize_scale,
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    recognition_result = recognition_session.recognize(rgb_small_frame)

    for face_match in recognition_result.matches:
        top, right, bottom, left = face_match.location
        name = face_match.name
        inv_scale = 1.0 / frame_resize_scale
        top = int(top * inv_scale)
        right = int(right * inv_scale)
        bottom = int(bottom * inv_scale)
        left = int(left * inv_scale)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{name} ({face_match.distance:.2f})"
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

    return frame, list(recognition_result.recognized_names)


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

    # Adapter callbacks resolve the module-level functions when invoked. This
    # keeps the application's existing seams replaceable in hardware-free tests.
    recognizer = FaceRecognitionAdapter(
        detector=lambda image, model: detect_faces(image, model=model),
        encoder=lambda image, locations: encode_faces(image, locations),
        matcher=lambda encoding, known, labels, tolerance: match_face(
            encoding, known, labels, tolerance=tolerance
        ),
    )
    image_store = OpenCVImageStore(known_faces_folder)
    recognition_service = RecognitionService(
        recognizer,
        image_store,
        attendance_system,
        detection_model,
        float(min_confidence),
    )
    enrollment_service = EnrollmentService(
        recognizer,
        image_store,
        SystemClock(now_provider=lambda: datetime.now()),
        TimestampEnrollmentIdGenerator(),
        detection_model,
    )

    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

    # Store config in app context so routes can use it
    app.config["DETECTION_MODEL"] = detection_model
    app.config["MIN_CONFIDENCE"] = float(min_confidence)
    app.config["FRAME_RESIZE_SCALE"] = float(frame_resize_scale)
    app.config["ATTENDANCE_PATH"] = attendance_path
    app.config["ATTENDANCE_SYSTEM"] = attendance_system
    app.config["KNOWN_FACES_FOLDER"] = known_faces_folder
    app.config["IMAGE_STORE"] = image_store
    app.config["RECOGNIZER"] = recognizer
    app.config["RECOGNITION_SERVICE"] = recognition_service
    app.config["ENROLLMENT_SERVICE"] = enrollment_service

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
        recognition_session = app.config["RECOGNITION_SERVICE"].load_session()

        try:
            while True:
                success, frame = video_capture.read()
                if not success:
                    break

                frame, _recognized_names = _recognize_frame(
                    frame,
                    recognition_session,
                    app.config["FRAME_RESIZE_SCALE"],
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

        recognition_session = app.config["RECOGNITION_SERVICE"].load_session()
        annotated_frame, recognized_names = _recognize_frame(
            frame,
            recognition_session,
            app.config["FRAME_RESIZE_SCALE"],
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

        person_name = _safe_person_name(name)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = app.config["ENROLLMENT_SERVICE"].enroll(
            person_name,
            rgb_frame,
            frame,
        )
        if result.status is EnrollmentStatus.NO_FACE:
            return jsonify({"error": "No face detected"}), 400
        if result.status is EnrollmentStatus.SAVE_FAILED:
            return jsonify({"error": "Failed to save enrollment image"}), 500

        return jsonify(
            {
                "success": True,
                "message": f"Successfully enrolled {result.person_name}",
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
        persons = app.config["IMAGE_STORE"].list_people()
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
