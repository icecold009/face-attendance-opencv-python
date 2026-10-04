"""Recognition orchestration without web-framework or camera dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .ports import (
    AttendanceRepository,
    FaceLocation,
    FaceRecognizer,
    ImageStore,
)


@dataclass(frozen=True)
class FaceMatch:
    location: FaceLocation
    name: str
    distance: float


@dataclass(frozen=True)
class RecognitionResult:
    matches: tuple[FaceMatch, ...]
    recognized_names: tuple[str, ...]


def _stack_encodings(encodings: Sequence[np.ndarray]) -> np.ndarray:
    if not encodings:
        return np.empty((0, 128), dtype=np.float32)
    return np.stack(
        [np.asarray(encoding, dtype=np.float32) for encoding in encodings],
        axis=0,
    )


class RecognitionSession:
    """A stable set of known faces used for one request or one video stream."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        attendance_repository: AttendanceRepository,
        known_encodings: np.ndarray,
        known_labels: Sequence[str],
        detection_model: str,
        tolerance: float,
    ) -> None:
        self._recognizer = recognizer
        self._attendance_repository = attendance_repository
        self._known_encodings = np.asarray(known_encodings, dtype=np.float32)
        self._known_labels = tuple(known_labels)
        self._detection_model = detection_model
        self._tolerance = tolerance

    def recognize(self, rgb_image: np.ndarray) -> RecognitionResult:
        face_locations = self._recognizer.detect_faces(
            rgb_image, model=self._detection_model
        )
        face_encodings = self._recognizer.encode_faces(
            rgb_image, face_locations
        )
        matches: list[FaceMatch] = []
        recognized_names: list[str] = []

        for face_encoding, location in zip(face_encodings, face_locations):
            if self._known_encodings.size == 0:
                name, distance = "Unknown", 1.0
            else:
                name, distance = self._recognizer.match_face(
                    face_encoding,
                    self._known_encodings,
                    self._known_labels,
                    tolerance=self._tolerance,
                )

            matches.append(FaceMatch(location, name, float(distance)))
            if name != "Unknown":
                recognized_names.append(name)
                # Attendance failure has never changed the recognition result.
                self._attendance_repository.mark_attendance(name)

        return RecognitionResult(tuple(matches), tuple(recognized_names))


class RecognitionService:
    def __init__(
        self,
        recognizer: FaceRecognizer,
        image_store: ImageStore,
        attendance_repository: AttendanceRepository,
        detection_model: str,
        tolerance: float,
    ) -> None:
        self._recognizer = recognizer
        self._image_store = image_store
        self._attendance_repository = attendance_repository
        self._detection_model = detection_model
        self._tolerance = tolerance

    def load_session(self) -> RecognitionSession:
        known_encodings: list[np.ndarray] = []
        known_labels: list[str] = []
        for known_face in self._image_store.iter_known_face_images():
            face_locations = self._recognizer.detect_faces(
                known_face.rgb_image, model=self._detection_model
            )
            face_encodings = self._recognizer.encode_faces(
                known_face.rgb_image, face_locations
            )
            if not face_encodings:
                continue
            known_encodings.append(face_encodings[0])
            known_labels.append(known_face.person_name)

        return RecognitionSession(
            self._recognizer,
            self._attendance_repository,
            _stack_encodings(known_encodings),
            known_labels,
            self._detection_model,
            self._tolerance,
        )
