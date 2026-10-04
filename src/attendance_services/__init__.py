"""Framework and hardware independent attendance application services."""

from .enrollment import (
    EnrollmentResult,
    EnrollmentService,
    EnrollmentStatus,
    normalize_person_name,
)
from .recognition import (
    FaceMatch,
    RecognitionResult,
    RecognitionService,
    RecognitionSession,
)

__all__ = [
    "EnrollmentResult",
    "EnrollmentService",
    "EnrollmentStatus",
    "FaceMatch",
    "RecognitionResult",
    "RecognitionService",
    "RecognitionSession",
    "normalize_person_name",
]
