"""Enrollment decisions and outcomes without Flask or OpenCV dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from .ports import Clock, EnrollmentIdGenerator, FaceRecognizer, ImageStore


class EnrollmentStatus(str, Enum):
    ENROLLED = "enrolled"
    NO_FACE = "no_face"
    SAVE_FAILED = "save_failed"


@dataclass(frozen=True)
class EnrollmentResult:
    status: EnrollmentStatus
    person_name: str


def normalize_person_name(name: str) -> str | None:
    normalized = name.strip()
    if not normalized or normalized in {".", ".."}:
        return None
    if "/" in normalized or "\\" in normalized:
        return None
    return normalized


class TimestampEnrollmentIdGenerator:
    def filename_for(self, timestamp: datetime) -> str:
        return timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"


class EnrollmentService:
    def __init__(
        self,
        recognizer: FaceRecognizer,
        image_store: ImageStore,
        clock: Clock,
        id_generator: EnrollmentIdGenerator,
        detection_model: str,
    ) -> None:
        self._recognizer = recognizer
        self._image_store = image_store
        self._clock = clock
        self._id_generator = id_generator
        self._detection_model = detection_model

    def enroll(
        self,
        person_name: str,
        rgb_image: np.ndarray,
        bgr_image: np.ndarray,
    ) -> EnrollmentResult:
        face_locations = self._recognizer.detect_faces(
            rgb_image, model=self._detection_model
        )
        if not self._recognizer.encode_faces(rgb_image, face_locations):
            return EnrollmentResult(EnrollmentStatus.NO_FACE, person_name)

        filename = self._id_generator.filename_for(self._clock.now())
        if not self._image_store.save_enrollment_image(
            person_name, bgr_image, filename
        ):
            return EnrollmentResult(EnrollmentStatus.SAVE_FAILED, person_name)
        return EnrollmentResult(EnrollmentStatus.ENROLLED, person_name)
