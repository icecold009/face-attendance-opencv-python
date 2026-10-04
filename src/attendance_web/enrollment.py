"""Enrollment routes and their compatibility response mapping."""

import cv2
from flask import Blueprint, current_app, jsonify, request

from attendance_services.enrollment import EnrollmentStatus, normalize_person_name

from .common import decode_frame


enrollment_blueprint = Blueprint("enrollment", __name__)


@enrollment_blueprint.post("/enroll")
def enroll():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    frame_b64 = data.get("frame")
    if not isinstance(name, str) or normalize_person_name(name) is None:
        return jsonify({"error": "A valid name is required"}), 400
    if not isinstance(frame_b64, str) or not frame_b64:
        return jsonify({"error": "No frame provided"}), 400

    frame = decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "Failed to decode frame"}), 400

    person_name = normalize_person_name(name)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = current_app.config["ENROLLMENT_SERVICE"].enroll(
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
