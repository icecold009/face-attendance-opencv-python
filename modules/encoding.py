# encoding.py
import face_recognition

def encode_faces(rgb_frame, face_locations):
    """
    Compute face encodings for detected faces.

    Args:
        rgb_frame: np.ndarray, image in RGB format.
        face_locations: list of face bounding boxes.

    Returns:
        List of 128-d encodings (numpy arrays).
    """
    return face_recognition.face_encodings(rgb_frame, face_locations)