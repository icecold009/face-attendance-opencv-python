"""Unit tests for the OpenCV-backed frame source lifecycle."""

import numpy as np

import frame_source


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = iter(frames)
        self.opened = opened
        self.read_count = 0
        self.release_count = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.read_count += 1
        return next(self.frames, (False, None))

    def release(self):
        self.release_count += 1


def test_opencv_frame_source_opens_lazily_and_releases_once(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame)])
    opened_indices = []

    def open_capture(index):
        opened_indices.append(index)
        return capture

    monkeypatch.setattr(frame_source.cv2, "VideoCapture", open_capture)
    source = frame_source.OpenCVFrameSource(camera_index=3)

    assert opened_indices == []
    with source as opened_source:
        assert opened_source is source
        success, actual_frame = source.read()
        assert success is True
        assert actual_frame is frame
        source.release()

    assert opened_indices == [3]
    assert capture.read_count == 1
    assert capture.release_count == 1


def test_opencv_frame_source_releases_when_camera_did_not_open(monkeypatch):
    capture = FakeCapture([], opened=False)
    monkeypatch.setattr(frame_source.cv2, "VideoCapture", lambda _index: capture)

    with frame_source.OpenCVFrameSource() as source:
        assert source.read() == (False, None)

    assert capture.read_count == 0
    assert capture.release_count == 1


def test_opencv_frame_source_releases_after_read_failure(monkeypatch):
    capture = FakeCapture([])
    monkeypatch.setattr(frame_source.cv2, "VideoCapture", lambda _index: capture)

    with frame_source.OpenCVFrameSource() as source:
        assert source.read() == (False, None)

    assert capture.read_count == 1
    assert capture.release_count == 1
