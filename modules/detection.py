# detection.py
import face_recognition

def detect_faces(rgb_frame, model="hog"):
    """
    Detect faces in an RGB frame.

    Args:
        rgb_frame: np.ndarray, image in RGB format.
        model: str, detection model ('hog' or 'cnn').

    Returns:
        List of (top, right, bottom, left) face locations.
    """
    return face_recognition.face_locations(rgb_frame, model=model)