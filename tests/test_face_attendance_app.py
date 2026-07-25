"""Unit tests for the Flask application factory."""

from pathlib import Path

import cv2
import pytest
from flask import Flask

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
