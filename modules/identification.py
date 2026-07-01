import numpy as np
from typing import List, Tuple

def match_face(
    face_encoding,
    known_encodings,
    known_labels: List[str],
    tolerance: float = 0.6,
) -> Tuple[str, float]:
    """
    Match a single face encoding against known encodings.

    `known_encodings` can be a list of arrays or a 2D numpy array.
    """
    enc_array = np.asarray(known_encodings, dtype=np.float32)
    if enc_array.ndim == 1:
        # single encoding case: reshape to (1, 128)
        enc_array = enc_array.reshape(1, -1)

    # Defensive: no known encodings
    if enc_array.shape[0] == 0:
        return "Unknown", 1.0

    distances = np.linalg.norm(enc_array - face_encoding, axis=1)
    min_idx = int(np.argmin(distances))
    min_dist = float(distances[min_idx])

    if min_dist <= tolerance:
        return known_labels[min_idx], min_dist
    return "Unknown", min_dist