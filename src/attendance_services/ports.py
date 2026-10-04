"""Typed boundaries used by the attendance services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol, Sequence

import numpy as np


FaceLocation = tuple[int, int, int, int]
FaceEncoding = np.ndarray


@dataclass(frozen=True)
class KnownFaceImage:
    """An RGB image loaded for one enrolled person."""

    person_name: str
    rgb_image: np.ndarray


class FaceRecognizer(Protocol):
    def detect_faces(
        self, rgb_image: np.ndarray, model: str
    ) -> Sequence[FaceLocation]: ...

    def encode_faces(
        self,
        rgb_image: np.ndarray,
        face_locations: Sequence[FaceLocation],
    ) -> Sequence[FaceEncoding]: ...

    def match_face(
        self,
        face_encoding: FaceEncoding,
        known_encodings: np.ndarray,
        known_labels: Sequence[str],
        tolerance: float,
    ) -> tuple[str, float]: ...


class AttendanceRepository(Protocol):
    def mark_attendance(self, name: str) -> bool: ...


class ImageStore(Protocol):
    def iter_known_face_images(self) -> Iterable[KnownFaceImage]: ...

    def save_enrollment_image(
        self, person_name: str, bgr_image: np.ndarray, filename: str
    ) -> bool: ...

    def list_people(self) -> list[str]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class EnrollmentIdGenerator(Protocol):
    def filename_for(self, timestamp: datetime) -> str: ...
