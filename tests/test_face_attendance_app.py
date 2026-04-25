"""Unit tests for FaceAttendanceApp (no camera / face_recognition required)."""

import os
import sys
import pytest

# Ensure src/ is importable (conftest.py does this, but be explicit for clarity)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_attendance_app import FaceAttendanceApp


@pytest.fixture
def app(tmp_path):
    """Return a FaceAttendanceApp with empty temporary directories."""
    enrollment_dir = tmp_path / 'images'
    attendance_dir = tmp_path / 'attendance'
    return FaceAttendanceApp(
        enrollment_path=str(enrollment_dir),
        attendance_path=str(attendance_dir),
    )


# ---------------------------------------------------------------------------
# get_enrolled_persons
# ---------------------------------------------------------------------------

def test_get_enrolled_persons_empty(app):
    persons = app.get_enrolled_persons()
    assert persons == []


def test_get_enrolled_persons_returns_list(app):
    assert isinstance(app.get_enrolled_persons(), list)


# ---------------------------------------------------------------------------
# get_attendance_today
# ---------------------------------------------------------------------------

def test_get_attendance_today_empty(app):
    records = app.get_attendance_today()
    assert records == []


def test_get_attendance_today_returns_list(app):
    assert isinstance(app.get_attendance_today(), list)


# ---------------------------------------------------------------------------
# load_and_encode_faces — directory creation side-effects
# ---------------------------------------------------------------------------

def test_load_creates_enrollment_directory(tmp_path):
    enrollment_dir = tmp_path / 'new_enrollment'
    app = FaceAttendanceApp(
        enrollment_path=str(enrollment_dir),
        attendance_path=str(tmp_path / 'attendance'),
    )
    assert enrollment_dir.exists()


def test_reload_does_not_raise_on_empty_dir(app):
    """Calling load_and_encode_faces on an empty dir should not raise."""
    app.load_and_encode_faces()
    assert app.known_face_encodings == []
    assert app.known_face_names == []
