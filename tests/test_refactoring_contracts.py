"""Behavioral contracts captured before extracting attendance services and routes."""

import base64
import socket
from datetime import datetime as RealDateTime

import cv2
import numpy as np
import pandas as pd
import pytest

import attendance as attendance_module
import face_attendance_app as app_module
from attendance_web import attendance as attendance_routes
from attendance_web import dashboard as dashboard_routes
from attendance_web import recognition as recognition_routes
from attendance_services.ports import KnownFaceImage
from modules.identification import match_face


class FrozenDateTime:
    @classmethod
    def now(cls):
        return RealDateTime(2025, 4, 3, 2, 1, 5, 6789)


@pytest.fixture(autouse=True)
def forbid_hardware_and_network(monkeypatch):
    def unexpected_io(*_args, **_kwargs):
        pytest.fail("characterization tests must not use hardware or network")

    monkeypatch.setattr(app_module.cv2, "VideoCapture", unexpected_io)
    monkeypatch.setattr(socket.socket, "connect", unexpected_io)
    monkeypatch.setattr(socket.socket, "connect_ex", unexpected_io)
    monkeypatch.setattr(socket, "create_connection", unexpected_io)


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "BASE_DIR", tmp_path)
    return app_module.create_app()


def jpeg_base64():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", frame)
    assert success
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def provide_known_faces(app, monkeypatch, labels):
    samples = [
        KnownFaceImage(label, np.zeros((4, 4, 3), dtype=np.uint8))
        for label in labels
    ]
    monkeypatch.setattr(
        app.config["IMAGE_STORE"],
        "iter_known_face_images",
        lambda: iter(samples),
    )


def test_routes_keep_their_current_http_methods(isolated_app):
    routes = {
        rule.rule: {
            method
            for method in rule.methods
            if method not in {"HEAD", "OPTIONS"}
        }
        for rule in isolated_app.url_map.iter_rules()
    }

    assert routes["/"] == {"GET"}
    assert routes["/video_feed"] == {"GET"}
    assert routes["/recognize"] == {"POST"}
    assert routes["/enroll"] == {"POST"}
    assert routes["/attendance"] == {"GET"}
    assert routes["/enrolled-persons"] == {"GET"}
    assert routes["/health"] == {"GET"}


def test_index_keeps_primary_dashboard_controls():
    response = app_module.create_app().test_client().get("/")

    assert response.status_code == 200
    for marker in (
        b"<title>Face Attendance System</title>",
        b'id="startBtn"',
        b"Start Recognition",
        b'id="enrollBtn"',
        b"Enroll New Person",
        b'id="attendanceBtn"',
        b"View Attendance",
    ):
        assert marker in response.data


def test_health_response_keeps_timestamp_format(isolated_app, monkeypatch):
    monkeypatch.setattr(dashboard_routes, "datetime", FrozenDateTime)

    response = isolated_app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "timestamp": "2025-04-03T02:01:05.006789",
    }


def test_recognize_returns_multiple_known_faces_and_marks_duplicates_once(
    isolated_app, monkeypatch
):
    provide_known_faces(isolated_app, monkeypatch, ["Alice"])
    monkeypatch.setattr(
        app_module,
        "detect_faces",
        lambda *_args, **_kwargs: [
            (0, 4, 4, 0),
            (0, 8, 4, 4),
            (4, 4, 8, 0),
        ],
    )
    monkeypatch.setattr(
        app_module,
        "encode_faces",
        lambda *_args: [
            np.zeros(128, dtype=np.float32),
            np.ones(128, dtype=np.float32) * 10,
            np.zeros(128, dtype=np.float32),
        ],
    )
    monkeypatch.setattr(recognition_routes, "datetime", FrozenDateTime)

    response = isolated_app.test_client().post(
        "/recognize",
        json={"frame": "data:image/jpeg;base64," + jpeg_base64()},
    )

    payload = response.get_json()
    attendance = pd.read_csv(
        isolated_app.config["ATTENDANCE_SYSTEM"].get_attendance_file()
    )
    annotated = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(payload["annotated_frame"]),
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["recognized_names"] == ["Alice", "Alice"]
    assert payload["timestamp"] == "2025-04-03 02:01:05"
    assert annotated is not None
    assert attendance["Name"].tolist() == ["Alice"]


def test_recognize_no_faces_returns_empty_names(isolated_app, monkeypatch):
    provide_known_faces(isolated_app, monkeypatch, [])
    monkeypatch.setattr(
        app_module,
        "detect_faces",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(app_module, "encode_faces", lambda *_args: [])

    response = isolated_app.test_client().post(
        "/recognize",
        json={"frame": jpeg_base64()},
    )

    assert response.status_code == 200
    assert response.get_json()["recognized_names"] == []


def test_recognize_loads_a_fresh_known_face_session_per_request(
    isolated_app, monkeypatch
):
    loads = []
    monkeypatch.setattr(
        isolated_app.config["IMAGE_STORE"],
        "iter_known_face_images",
        lambda: loads.append("loaded") or iter(()),
    )
    monkeypatch.setattr(
        app_module, "detect_faces", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(app_module, "encode_faces", lambda *_args: [])
    client = isolated_app.test_client()

    responses = [
        client.post("/recognize", json={"frame": jpeg_base64()})
        for _ in range(2)
    ]

    assert [response.status_code for response in responses] == [200, 200]
    assert loads == ["loaded", "loaded"]


def test_recognize_malformed_base64_keeps_400_error(isolated_app):
    response = isolated_app.test_client().post(
        "/recognize",
        json={"frame": "not base64"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Failed to decode frame"}


def test_recognize_encode_failure_keeps_500_error(isolated_app, monkeypatch):
    frame = jpeg_base64()
    provide_known_faces(isolated_app, monkeypatch, [])
    monkeypatch.setattr(
        app_module, "detect_faces", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(app_module, "encode_faces", lambda *_args: [])
    monkeypatch.setattr(app_module.cv2, "imencode", lambda *_args: (False, None))

    response = isolated_app.test_client().post(
        "/recognize",
        json={"frame": frame},
    )

    assert response.status_code == 500
    assert response.get_json() == {"error": "Failed to encode result frame"}


@pytest.mark.parametrize("name", ["", "  ", ".", "..", "../outside", "a\\\\b"])
def test_enroll_rejects_invalid_names(isolated_app, name):
    response = isolated_app.test_client().post(
        "/enroll",
        json={"name": name, "frame": "unused"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "A valid name is required"}


def test_enroll_rejects_malformed_frame(isolated_app):
    response = isolated_app.test_client().post(
        "/enroll",
        json={"name": "Alice", "frame": "not base64"},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Failed to decode frame"}


def test_enroll_requires_a_detected_face(isolated_app, monkeypatch):
    monkeypatch.setattr(
        app_module, "detect_faces", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(app_module, "encode_faces", lambda *_args: [])

    response = isolated_app.test_client().post(
        "/enroll",
        json={"name": "Alice", "frame": jpeg_base64()},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "No face detected"}
    assert not (isolated_app.config["KNOWN_FACES_FOLDER"] / "Alice").exists()


def test_enroll_uses_existing_timestamped_jpeg_name(
    isolated_app, monkeypatch
):
    monkeypatch.setattr(
        app_module, "detect_faces", lambda *_args, **_kwargs: [(1, 9, 9, 1)]
    )
    monkeypatch.setattr(
        app_module,
        "encode_faces",
        lambda *_args: [np.zeros(128, dtype=np.float32)],
    )
    monkeypatch.setattr(app_module, "datetime", FrozenDateTime)

    response = isolated_app.test_client().post(
        "/enroll",
        json={"name": "Alice", "frame": jpeg_base64()},
    )

    files = list(
        (isolated_app.config["KNOWN_FACES_FOLDER"] / "Alice").glob("*.jpg")
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "success": True,
        "message": "Successfully enrolled Alice",
    }
    assert [path.name for path in files] == ["20250403_020105_006789.jpg"]


def test_enroll_preserves_unicode_name_in_path_and_response(
    isolated_app, monkeypatch
):
    monkeypatch.setattr(
        app_module, "detect_faces", lambda *_args, **_kwargs: [(1, 9, 9, 1)]
    )
    monkeypatch.setattr(
        app_module,
        "encode_faces",
        lambda *_args: [np.zeros(128, dtype=np.float32)],
    )
    monkeypatch.setattr(app_module, "datetime", FrozenDateTime)
    saved_paths = []
    monkeypatch.setattr(
        app_module.cv2,
        "imwrite",
        lambda path, _frame: saved_paths.append(path) or True,
    )
    person_name = "Zoë 東京"

    response = isolated_app.test_client().post(
        "/enroll",
        json={"name": f"  {person_name}  ", "frame": jpeg_base64()},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "success": True,
        "message": f"Successfully enrolled {person_name}",
    }
    assert saved_paths == [
        str(
            isolated_app.config["KNOWN_FACES_FOLDER"]
            / person_name
            / "20250403_020105_006789.jpg"
        )
    ]


def test_enroll_save_failure_keeps_500_response(isolated_app, monkeypatch):
    monkeypatch.setattr(
        app_module, "detect_faces", lambda *_args, **_kwargs: [(1, 9, 9, 1)]
    )
    monkeypatch.setattr(
        app_module,
        "encode_faces",
        lambda *_args: [np.zeros(128, dtype=np.float32)],
    )
    monkeypatch.setattr(app_module.cv2, "imwrite", lambda *_args: False)

    response = isolated_app.test_client().post(
        "/enroll",
        json={"name": "Alice", "frame": jpeg_base64()},
    )

    assert response.status_code == 500
    assert response.get_json() == {"error": "Failed to save enrollment image"}


def test_attendance_returns_empty_list_when_storage_summary_fails(
    isolated_app, monkeypatch
):
    monkeypatch.setattr(attendance_routes, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        isolated_app.config["ATTENDANCE_SYSTEM"],
        "get_attendance_summary",
        lambda: None,
    )

    response = isolated_app.test_client().get("/attendance")

    assert response.status_code == 200
    assert response.get_json() == {
        "success": True,
        "attendance": [],
        "date": "2025-04-03",
        "count": 0,
    }


def test_attendance_csv_freezes_date_timestamp_and_column_order(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(attendance_module, "get_date", lambda: "2025-04-03")
    monkeypatch.setattr(
        attendance_module, "get_timestamp", lambda: "02:01:05"
    )
    system = attendance_module.AttendanceSystem(str(tmp_path))

    person_name = "Zoë 東京"
    assert system.mark_attendance(person_name) is True
    path = tmp_path / "Attendance_2025-04-03.csv"
    saved = pd.read_csv(path)

    assert list(saved.columns) == ["Name", "Time", "Status"]
    assert saved.to_dict(orient="records") == [
        {"Name": person_name, "Time": "02:01:05", "Status": "Present"}
    ]


def test_attendance_storage_error_does_not_mark_person(
    tmp_path, monkeypatch
):
    system = attendance_module.AttendanceSystem(str(tmp_path))

    def fail_read(*_args, **_kwargs):
        raise OSError("simulated storage read failure")

    monkeypatch.setattr(attendance_module.pd, "read_csv", fail_read)

    assert system.mark_attendance("Alice") is False
    assert system.marked_today == set()


def test_match_face_keeps_first_label_for_equal_distance():
    encoding = np.zeros(128, dtype=np.float32)
    known_encodings = np.stack(
        [
            np.ones(128, dtype=np.float32),
            -np.ones(128, dtype=np.float32),
        ]
    )

    name, distance = match_face(
        encoding,
        known_encodings,
        ["First", "Second"],
        tolerance=20,
    )

    assert name == "First"
    assert distance == pytest.approx(np.sqrt(128))


def test_match_face_accepts_distance_at_tolerance_boundary():
    encoding = np.zeros(128, dtype=np.float32)
    encoding[0] = 0.5

    name, distance = match_face(
        encoding,
        np.zeros((1, 128), dtype=np.float32),
        ["Alice"],
        tolerance=0.5,
    )

    assert name == "Alice"
    assert distance == pytest.approx(0.5)


def test_match_face_keeps_unknown_result_above_tolerance():
    name, distance = match_face(
        np.ones(128, dtype=np.float32),
        np.zeros((1, 128), dtype=np.float32),
        ["Alice"],
        tolerance=0.6,
    )

    assert name == "Unknown"
    assert distance == pytest.approx(np.sqrt(128))


def test_video_feed_releases_camera_after_normal_end(isolated_app, monkeypatch):
    class FakeVideoCapture:
        def __init__(self):
            self.releases = 0

        def read(self):
            return False, None

        def release(self):
            self.releases += 1

    capture = FakeVideoCapture()
    destroyed = []
    monkeypatch.setattr(
        app_module.cv2, "VideoCapture", lambda _index: capture
    )
    monkeypatch.setattr(
        app_module.cv2, "destroyAllWindows", lambda: destroyed.append(True)
    )
    provide_known_faces(isolated_app, monkeypatch, [])

    response = isolated_app.test_client().get("/video_feed", buffered=False)
    body = list(response.response)
    response.close()

    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    assert body == []
    assert capture.releases == 1
    assert destroyed == [True]


def test_video_feed_loads_known_faces_once_for_the_stream(
    isolated_app, monkeypatch
):
    class FakeVideoCapture:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            return (self.reads <= 3, np.zeros((40, 40, 3), dtype=np.uint8))

        def release(self):
            pass

    capture = FakeVideoCapture()
    session_loads = []
    monkeypatch.setattr(
        app_module.cv2, "VideoCapture", lambda _index: capture
    )
    monkeypatch.setattr(app_module.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        isolated_app.config["RECOGNITION_SERVICE"],
        "load_session",
        lambda: session_loads.append("loaded") or object(),
    )
    monkeypatch.setattr(
        recognition_routes,
        "recognize_frame",
        lambda frame, _session, _scale: (frame, []),
    )

    response = isolated_app.test_client().get("/video_feed", buffered=False)
    list(response.response)
    response.close()

    assert session_loads == ["loaded"]
    assert capture.reads == 4


def test_video_feed_releases_camera_when_frame_processing_raises(
    isolated_app, monkeypatch
):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    class FakeVideoCapture:
        def __init__(self):
            self.releases = 0
            self.reads = 0

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, frame.copy()
            return False, None

        def release(self):
            self.releases += 1

    capture = FakeVideoCapture()
    destroyed = []
    monkeypatch.setattr(
        app_module.cv2, "VideoCapture", lambda _index: capture
    )
    monkeypatch.setattr(
        app_module.cv2, "destroyAllWindows", lambda: destroyed.append(True)
    )
    provide_known_faces(isolated_app, monkeypatch, [])

    def fail_processing(*_args, **_kwargs):
        raise RuntimeError("simulated frame processing failure")

    monkeypatch.setattr(
        recognition_routes, "recognize_frame", fail_processing
    )

    with pytest.raises(RuntimeError, match="simulated frame processing failure"):
        isolated_app.test_client().get("/video_feed", buffered=False)

    assert capture.releases == 1
    assert destroyed == [True]
