"""Focused tests for the framework-independent attendance services."""

import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from attendance_adapters import OpenCVImageStore
from attendance_services.enrollment import (
    EnrollmentResult,
    EnrollmentService,
    EnrollmentStatus,
    TimestampEnrollmentIdGenerator,
    normalize_person_name,
)
from attendance_services.ports import KnownFaceImage
from attendance_services.recognition import RecognitionService


class FakeImageStore:
    def __init__(self, images=(), save_result=True):
        self.images = list(images)
        self.save_result = save_result
        self.saved = []

    def iter_known_face_images(self):
        return iter(self.images)

    def save_enrollment_image(self, person_name, bgr_image, filename):
        self.saved.append((person_name, bgr_image.copy(), filename))
        return self.save_result

    def list_people(self):
        return []


class FakeAttendance:
    def __init__(self, result=False):
        self.result = result
        self.names = []

    def mark_attendance(self, name):
        self.names.append(name)
        return self.result


class FakeRecognizer:
    def __init__(self):
        self.match_calls = []

    def detect_faces(self, rgb_image, model):
        marker = int(rgb_image[0, 0, 0])
        if marker == 4:
            return [(0, 4, 4, 0), (1, 5, 5, 1)]
        return [(0, 2, 2, 0)]

    def encode_faces(self, rgb_image, face_locations):
        marker = int(rgb_image[0, 0, 0])
        if marker == 2:
            return []
        if marker == 3:
            return [np.full(128, 3), np.full(128, 30)]
        if marker == 4:
            return [np.zeros(128), np.ones(128)]
        return [np.full(128, marker)]

    def match_face(self, face_encoding, known_encodings, known_labels, tolerance):
        self.match_calls.append((known_encodings.copy(), list(known_labels), tolerance))
        if float(face_encoding[0]) == 0:
            return "Alice", 0.2
        return "Unknown", 0.8


class FrozenClock:
    def now(self):
        return datetime(2025, 4, 3, 2, 1, 5, 6789)


class FakeIds:
    def filename_for(self, timestamp):
        return TimestampEnrollmentIdGenerator().filename_for(timestamp)


def image(marker):
    return np.full((4, 4, 3), marker, dtype=np.uint8)


def test_service_package_does_not_import_flask_or_opencv():
    import attendance_services

    service_dir = Path(attendance_services.__file__).parent
    forbidden = {"flask", "cv2"}
    imports = set()
    for source_path in service_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    assert imports.isdisjoint(forbidden)


def test_recognition_loads_first_encoding_in_image_order_and_keeps_partial_success():
    store = FakeImageStore(
        [
            KnownFaceImage("Alice", image(1)),
            KnownFaceImage("NoEncoding", image(2)),
            KnownFaceImage("Carol", image(3)),
        ]
    )
    attendance = FakeAttendance(result=False)
    recognizer = FakeRecognizer()
    service = RecognitionService(recognizer, store, attendance, "hog", 0.6)

    session = service.load_session()
    result = session.recognize(image(4))

    assert session._known_labels == ("Alice", "Carol")
    assert session._known_encodings.shape == (2, 128)
    assert session._known_encodings[:, 0].tolist() == [1.0, 3.0]
    assert result.recognized_names == ("Alice",)
    assert [match.name for match in result.matches] == ["Alice", "Unknown"]
    assert attendance.names == ["Alice"]
    assert recognizer.match_calls[0][1:] == (["Alice", "Carol"], 0.6)


def test_recognition_without_known_faces_returns_unknown_and_does_not_mark():
    attendance = FakeAttendance()
    service = RecognitionService(
        FakeRecognizer(), FakeImageStore(), attendance, "hog", 0.6
    )

    result = service.load_session().recognize(image(4))

    assert [match.name for match in result.matches] == ["Unknown", "Unknown"]
    assert [match.distance for match in result.matches] == [1.0, 1.0]
    assert result.recognized_names == ()
    assert attendance.names == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Zoë 東京 ", "Zoë 東京"),
        ("", None),
        ("  ", None),
        (".", None),
        ("..", None),
        ("../person", None),
        ("a\\b", None),
    ],
)
def test_person_name_normalization(raw, expected):
    assert normalize_person_name(raw) == expected


def test_enrollment_service_preserves_unicode_image_and_timestamped_name():
    store = FakeImageStore()
    service = EnrollmentService(
        FakeRecognizer(),
        store,
        FrozenClock(),
        FakeIds(),
        "hog",
    )
    original = image(9)

    result = service.enroll("Zoë 東京", image(1), original)

    assert result == EnrollmentResult(EnrollmentStatus.ENROLLED, "Zoë 東京")
    assert store.saved[0][0] == "Zoë 東京"
    np.testing.assert_array_equal(store.saved[0][1], original)
    assert store.saved[0][2] == "20250403_020105_006789.jpg"


def test_enrollment_service_reports_no_face_without_writing():
    store = FakeImageStore()
    service = EnrollmentService(
        FakeRecognizer(), store, FrozenClock(), FakeIds(), "hog"
    )

    result = service.enroll("Alice", image(2), image(9))

    assert result.status is EnrollmentStatus.NO_FACE
    assert store.saved == []


def test_enrollment_service_reports_image_store_failure():
    store = FakeImageStore(save_result=False)
    service = EnrollmentService(
        FakeRecognizer(), store, FrozenClock(), FakeIds(), "hog"
    )

    result = service.enroll("Alice", image(1), image(9))

    assert result.status is EnrollmentStatus.SAVE_FAILED
    assert len(store.saved) == 1


def test_opencv_image_store_skips_non_directories_and_unreadable_images(
    tmp_path, monkeypatch
):
    root = tmp_path / "people"
    (root / "Alice").mkdir(parents=True)
    (root / "Bob").mkdir()
    (root / "not-a-person.txt").write_text("ignored", encoding="utf-8")
    listings = {
        str(root): ["Alice", "not-a-person.txt", "Bob"],
        str(root / "Alice"): ["first.jpg", "broken.jpg"],
        str(root / "Bob"): ["last.jpg"],
    }
    monkeypatch.setattr(
        "attendance_adapters.os.listdir", lambda path: listings[str(path)]
    )
    monkeypatch.setattr(
        "attendance_adapters.cv2.imread",
        lambda path: None if path.endswith("broken.jpg") else image(1),
    )
    monkeypatch.setattr(
        "attendance_adapters.cv2.cvtColor",
        lambda pixels, _conversion: pixels + 1,
    )

    samples = list(OpenCVImageStore(root).iter_known_face_images())

    assert [
        (sample.person_name, int(sample.rgb_image[0, 0, 0]))
        for sample in samples
    ] == [("Alice", 2), ("Bob", 2)]
