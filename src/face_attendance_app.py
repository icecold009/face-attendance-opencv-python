"""Application factory and compatibility helpers for the Flask app."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask


BASE_DIR = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, BASE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


from attendance import AttendanceSystem
from attendance_adapters import FaceRecognitionAdapter, OpenCVImageStore, SystemClock
from attendance_services.enrollment import (
    EnrollmentService,
    TimestampEnrollmentIdGenerator,
    normalize_person_name,
)
from attendance_services.recognition import RecognitionService
from attendance_web import register_blueprints
from attendance_web.common import decode_frame, recognize_frame
from config import load_config

def detect_faces(rgb_image, model="hog"):
    """Load the face detector only when an operation needs it."""
    from modules.detection import detect_faces as implementation

    return implementation(rgb_image, model=model)


def encode_faces(rgb_image, face_locations):
    """Load the face encoder only when an operation needs it."""
    from modules.encoding import encode_faces as implementation

    return implementation(rgb_image, face_locations)


def match_face(face_encoding, known_encodings, known_labels, tolerance=0.6):
    """Load the face matcher only when an operation needs it."""
    from modules.identification import match_face as implementation

    return implementation(
        face_encoding,
        known_encodings,
        known_labels,
        tolerance=tolerance,
    )


# Retain the former helper names for callers that imported these private seams.
_decode_frame = decode_frame
_recognize_frame = recognize_frame
_safe_person_name = normalize_person_name


def create_app() -> Flask:
    """Create and wire the application without opening a camera or loading faces."""
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

    # Resolve app-module functions lazily so hardware-free tests can replace them.
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
    app.config.update(
        {
            "DETECTION_MODEL": detection_model,
            "MIN_CONFIDENCE": float(min_confidence),
            "FRAME_RESIZE_SCALE": float(frame_resize_scale),
            "ATTENDANCE_PATH": attendance_path,
            "ATTENDANCE_SYSTEM": attendance_system,
            "KNOWN_FACES_FOLDER": known_faces_folder,
            "IMAGE_STORE": image_store,
            "RECOGNIZER": recognizer,
            "RECOGNITION_SERVICE": recognition_service,
            "ENROLLMENT_SERVICE": enrollment_service,
        }
    )
    register_blueprints(app)
    return app


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
