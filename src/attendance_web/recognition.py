"""Live and single-frame recognition routes."""

from __future__ import annotations

import base64
from datetime import datetime

import cv2
from flask import Blueprint, Response, current_app, jsonify, request

from .common import decode_frame, recognize_frame


recognition_blueprint = Blueprint("recognition", __name__)


@recognition_blueprint.get("/video_feed")
def video_feed():
    recognition_service = current_app.config["RECOGNITION_SERVICE"]
    frame_resize_scale = current_app.config["FRAME_RESIZE_SCALE"]
    frame_source_factory = current_app.config["FRAME_SOURCE_FACTORY"]

    def generate_frames():
        try:
            with frame_source_factory() as frame_source:
                recognition_session = recognition_service.load_session()
                while True:
                    success, frame = frame_source.read()
                    if not success:
                        break

                    frame, _recognized_names = recognize_frame(
                        frame,
                        recognition_session,
                        frame_resize_scale,
                    )

                    ret, buffer = cv2.imencode(".jpg", frame)
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
        finally:
            cv2.destroyAllWindows()

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@recognition_blueprint.post("/recognize")
def recognize():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get("frame")
    if not isinstance(frame_b64, str) or not frame_b64:
        return jsonify({"error": "No frame provided"}), 400

    frame = decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "Failed to decode frame"}), 400

    recognition_session = current_app.config["RECOGNITION_SERVICE"].load_session()
    annotated_frame, recognized_names = recognize_frame(
        frame,
        recognition_session,
        current_app.config["FRAME_RESIZE_SCALE"],
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
