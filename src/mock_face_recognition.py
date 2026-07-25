"""Dependency-free fallback for environments without dlib.

This module provides a lightweight OpenCV-based face detection and encoding
implementation for local tests and constrained environments. It is an
explicit fallback and is not a replacement for the ``face_recognition``
package.
"""

import cv2
import numpy as np
from PIL import Image

# HOG-like encoding parameters
_HOG_SIZE   = 64   # face is resized to this before encoding
_HOG_GRID   = 5    # spatial grid (5x5 cells)
_HOG_BINS   = 8    # gradient orientation bins per cell → 5*5*8=200, trimmed to 128
_DISTANCE_NORMALIZATION = 1.09
# Face size filter: detections whose larger dimension exceeds this fraction of the
# smallest image dimension are discarded (removes multi-scale duplicates / oversized
# false positives from the Haar cascade that would produce incorrect encodings).
_MAX_FACE_RATIO = 0.4

def load_image_file(file_path):
    """Load an image file and return an RGB numpy array"""
    img = cv2.imread(str(file_path))
    if img is None:
        # Fallback to PIL for non-standard formats (e.g. PNG with alpha, WebP)
        pil_img = Image.open(file_path).convert("RGB")
        return np.array(pil_img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def face_locations(image, model='hog'):
    """
    Detect face locations using Haar cascade.
    Returns list of (top, right, bottom, left) tuples, sorted by face area (largest first).

    Detections whose larger side exceeds _MAX_FACE_RATIO * min(H, W) are filtered out
    to remove oversized false positives / multi-scale duplicates from the Haar cascade.
    If filtering would leave no faces, all detections are kept as a fallback.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    H, W   = gray.shape
    min_dim = min(H, W)
    max_size = _MAX_FACE_RATIO * min_dim

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = cascade.detectMultiScale(gray, 1.05, 4, minSize=(50, 50))

    if len(faces) == 0:
        return []

    # Discard oversized detections; fall back to all if none survive the filter
    filtered = [(x, y, w, h) for (x, y, w, h) in faces if max(w, h) < max_size]
    if not filtered:
        filtered = [(x, y, w, h) for (x, y, w, h) in faces]

    # Largest surviving face first
    filtered = sorted(filtered, key=lambda f: f[2] * f[3], reverse=True)
    return [(y, x + w, y + h, x) for (x, y, w, h) in filtered]

def face_encodings(image, face_locations_list=None, num_jitters=1):
    """
    Generate 128-D face encodings using gradient-orientation spatial histograms.
    Each encoding is a L2-normalised float32 vector.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if face_locations_list is None:
        face_locations_list = face_locations(image)

    encodings = []
    cell_size  = _HOG_SIZE // _HOG_GRID

    for (top, right, bottom, left) in face_locations_list:
        face_region = gray[int(top):int(bottom), int(left):int(right)]

        if face_region.size == 0:
            encodings.append(np.zeros(128, dtype=np.float32))
            continue

        # Resize to a fixed square and compute image gradients
        face_resized = cv2.resize(face_region, (_HOG_SIZE, _HOG_SIZE)).astype(np.float32)
        gx        = cv2.Sobel(face_resized, cv2.CV_32F, 1, 0, ksize=3)
        gy        = cv2.Sobel(face_resized, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle     = np.arctan2(gy, gx) * 180.0 / np.pi % 180.0  # 0–180°

        # Build gradient-orientation histogram for each grid cell
        raw_enc = []
        for row in range(_HOG_GRID):
            for col in range(_HOG_GRID):
                r0, r1 = row * cell_size, (row + 1) * cell_size
                c0, c1 = col * cell_size, (col + 1) * cell_size
                m = magnitude[r0:r1, c0:c1]
                a = angle[r0:r1, c0:c1]
                hist, _ = np.histogram(a, bins=_HOG_BINS, range=(0, 180), weights=m)
                cell_sum = hist.sum()
                raw_enc.extend(hist / cell_sum if cell_sum > 0 else np.zeros(_HOG_BINS, dtype=np.float32))

        # Trim to 128-D and L2-normalise
        encoding = np.array(raw_enc[:128], dtype=np.float32)
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding /= norm

        encodings.append(encoding)

    return encodings

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a face encoding against known face encodings.
    Returns list of booleans (True = match within tolerance).
    """
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    return list(distances <= tolerance)

def face_distance(face_encodings, face_to_compare):
    """
    Compare fallback encodings to a probe and return this module's distances.

    The cubic transform keeps the fallback's compressed HOG distances useful
    with the tolerance values used by the application.
    """
    if len(face_encodings) == 0:
        return np.array([])

    probe = np.array(face_to_compare, dtype=np.float32)
    raw = np.array(
        [np.linalg.norm(np.array(enc, dtype=np.float32) - probe)
         for enc in face_encodings],
        dtype=np.float32,
    )
    # Keep the fallback distance range useful for the application's tolerance.
    return raw ** 3 / _DISTANCE_NORMALIZATION


