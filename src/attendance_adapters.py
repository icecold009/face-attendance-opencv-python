"""OpenCV and application-specific implementations of service ports."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from attendance_services.ports import FaceLocation, KnownFaceImage
from modules.detection import detect_faces
from modules.encoding import encode_faces
from modules.identification import match_face


class FaceRecognitionAdapter:
    def __init__(
        self,
        detector: Callable[..., Sequence[FaceLocation]] | None = None,
        encoder: Callable[..., Sequence[np.ndarray]] | None = None,
        matcher: Callable[..., tuple[str, float]] | None = None,
    ) -> None:
        self._detector = detector or detect_faces
        self._encoder = encoder or encode_faces
        self._matcher = matcher or match_face

    def detect_faces(self, rgb_image: np.ndarray, model: str):
        return self._detector(rgb_image, model=model)

    def encode_faces(self, rgb_image: np.ndarray, face_locations):
        return self._encoder(rgb_image, face_locations)

    def match_face(self, face_encoding, known_encodings, known_labels, tolerance):
        return self._matcher(
            face_encoding,
            known_encodings,
            known_labels,
            tolerance=tolerance,
        )


class OpenCVImageStore:
    def __init__(self, known_faces_folder: Path) -> None:
        self._known_faces_folder = known_faces_folder

    def iter_known_face_images(self):
        folder_path = self._known_faces_folder
        if not folder_path.exists():
            return

        for person_name in os.listdir(folder_path):
            person_dir = folder_path / person_name
            if not person_dir.is_dir():
                continue
            for filename in os.listdir(person_dir):
                image_path = person_dir / filename
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield KnownFaceImage(person_name, rgb_image)

    def save_enrollment_image(
        self, person_name: str, bgr_image: np.ndarray, filename: str
    ) -> bool:
        person_dir = self._known_faces_folder / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(person_dir / filename), bgr_image))

    def list_people(self) -> list[str]:
        if not self._known_faces_folder.exists():
            return []
        return sorted(
            directory.name
            for directory in self._known_faces_folder.iterdir()
            if directory.is_dir()
        )


class SystemClock:
    def __init__(self, now_provider: Callable[[], datetime] | None = None) -> None:
        self._now_provider = now_provider or datetime.now

    def now(self) -> datetime:
        return self._now_provider()
