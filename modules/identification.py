# identification.py
import numpy as np
from typing import List, Tuple

def match_face(face_encoding,
               known_encodings: List[np.ndarray],
               known_labels: List[str],
               tolerance: float = 0.6) -> Tuple[str, float]:
    """
    Match a single face encoding against known encodings.

    Returns:
        (label, distance) where label is name/ID or 'Unknown'.
    """
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    min_idx = int(np.argmin(distances))
    min_dist = float(distances[min_idx])

    if min_dist <= tolerance:
        return known_labels[min_idx], min_dist
    return "Unknown", min_dist