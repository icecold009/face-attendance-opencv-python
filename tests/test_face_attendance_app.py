"""Unit tests for the Flask application factory."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from flask import Flask

import face_attendance_app as face_attendance_module
from face_attendance_app import create_app


@pytest.fixture
def app():
    """Return an application instance without opening the camera."""
    return create_app()


def test_create_app_returns_flask_application(app):
    assert isinstance(app, Flask)


def test_create_app_sets_expected_runtime_configuration(app):
    assert app.config["DETECTION_MODEL"] == "hog"
    assert app.config["MIN_CONFIDENCE"] == pytest.approx(0.6)
    assert app.config["FRAME_RESIZE_SCALE"] == pytest.approx(0.25)
    assert app.config["KNOWN_FACES_FOLDER"] == Path(__file__).parents[1] / "ImagesAttendance"
    assert app.config["ATTENDANCE_PATH"] == Path(__file__).parents[1] / "data" / "Attendance"


def test_index_route_returns_template(app):
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert response.content_type.startswith("text/html")


def test_video_feed_route_is_registered(app):
    routes = {rule.rule for rule in app.url_map.iter_rules()}

    assert "/video_feed" in routes


def test_create_app_does_not_open_camera(monkeypatch):
    def fail_if_camera_is_opened(*args, **kwargs):
        raise AssertionError("create_app() must not open the camera")

    monkeypatch.setattr(cv2, "VideoCapture", fail_if_camera_is_opened)

    app = create_app()

    assert isinstance(app, Flask)


def test_video_feed_marks_person_once_per_day(tmp_path, monkeypatch):
    """Repeated detections in the live feed create only one attendance row."""
    monkeypatch.setattr(face_attendance_module, "BASE_DIR", tmp_path)
    app = create_app()

    class FakeVideoCapture:
        def __init__(self):
            self.frames_read = 0

        def read(self):
            if self.frames_read >= 10:
                return False, None
            self.frames_read += 1
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(
        face_attendance_module.cv2,
        "VideoCapture",
        lambda _camera_index: FakeVideoCapture(),
    )
    monkeypatch.setattr(face_attendance_module.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        face_attendance_module,
        "_load_known_faces_from_folder",
        lambda *_args: ([np.zeros(128, dtype=np.float32)], ["Alice"]),
    )
    monkeypatch.setattr(
        face_attendance_module,
        "detect_faces",
        lambda *_args, **_kwargs: [(0, 2, 2, 0)],
    )
    monkeypatch.setattr(
        face_attendance_module,
        "encode_faces",
        lambda *_args: [np.zeros(128, dtype=np.float32)],
    )
    monkeypatch.setattr(
        face_attendance_module,
        "match_face",
        lambda *_args, **_kwargs: ("Alice", 0.2),
    )

    response = app.view_functions["video_feed"]()
    list(response.response)

    attendance_system = app.config["ATTENDANCE_SYSTEM"]
    attendance = pd.read_csv(attendance_system.get_attendance_file())
    assert len(attendance) == 1
    assert attendance.iloc[0]["Name"] == "Alice"
