"""Small Flask-adjacent helpers shared by route blueprints."""

from __future__ import annotations

import base64
from typing import List, Tuple

import cv2
import numpy as np

from attendance_services.recognition import RecognitionSession


def decode_frame(frame_b64: str) -> np.ndarray | None:
    if frame_b64.startswith("data:") and "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]
    try:
        frame_data = base64.b64decode(frame_b64, validate=True)
    except (ValueError, TypeError):
        return None

    return cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)


def recognize_frame(
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
