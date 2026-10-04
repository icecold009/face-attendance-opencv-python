"""Context-managed frame sources for the live recognition route."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def __enter__(self) -> "FrameSource": ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None: ...


class OpenCVFrameSource:
    """Own a camera capture from stream entry through one idempotent release."""

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._capture = None
        self._released = True

    def __enter__(self) -> "OpenCVFrameSource":
        if self._capture is not None and not self._released:
            raise RuntimeError("frame source is already open")
        self._capture = cv2.VideoCapture(self._camera_index)
        self._released = False
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None or self._released:
            raise RuntimeError("frame source is not open")
        if not self._capture.isOpened():
            return False, None
        return self._capture.read()

    def release(self) -> None:
        if self._capture is None or self._released:
            return
        self._released = True
        self._capture.release()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False
