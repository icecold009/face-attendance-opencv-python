import cv2
import numpy as np
import os
import face_recognition  # Local mock module
from PIL import Image
from utils import create_directory, save_pickle, load_pickle

class FaceRecognitionSystem:
    def __init__(self, encodings_path='encodings.pkl', tolerance=0.6):
        self.encodings_path = encodings_path
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()
    
    def encode_faces(self, image_path):
        """Encode faces in an image"""
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                return face_encodings[0]
            return None
        except Exception as e:
            print(f"Error encoding face from {image_path}: {e}")
            return None
    
    def add_person(self, name, image_path):
        """Add a new person to the system"""
        encoding = self.encode_faces(image_path)
        
        if encoding is not None:
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            self.save_encodings()
            return True
        return False
    
    def add_multiple_images(self, name, image_folder):
        """Add a person with multiple images"""
        if not os.path.exists(image_folder):
            return False
        
        added = False
        for image_name in os.listdir(image_folder):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, image_name)
                encoding = self.encode_faces(image_path)
                
                if encoding is not None:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    added = True
        
        if added:
            self.save_encodings()
        return added
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=self.tolerance
            )
            name = "Unknown"
            confidence = 0
            
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
            
            face_names.append((name, confidence))
        
        return face_locations, face_names
    
    def save_encodings(self):
        """Save encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        save_pickle(data, self.encodings_path)
    
    def load_encodings(self):
        """Load encodings from file"""
        data = load_pickle(self.encodings_path)
        
        if data:
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
    
    def draw_faces(self, frame, face_locations, face_names):
        """Draw rectangles and labels on frame"""
        for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame

if __name__ == "__main__":
    import os
    import sys
    # Path to test images
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    elon_image_path = os.path.join(base_path, "images", "Basic", "Elon musk.jpg")
    elon_test_path = os.path.join(base_path, "images", "Basic", "Elon test.jpg")
    bill_gates_path = os.path.join(base_path, "images", "Basic", "Bill gates.jpg")
    
    # Load images
    elon_image = face_recognition.load_image_file(elon_image_path)  # type: ignore
    elon_test = face_recognition.load_image_file(elon_test_path)  # type: ignore
    bill_gates_image = face_recognition.load_image_file(bill_gates_path)  # type: ignore
    
    # Convert BGR to RGB (face_recognition uses RGB)
    elon_image_rgb = cv2.cvtColor(cv2.imread(elon_image_path), cv2.COLOR_BGR2RGB)
    elon_test_rgb = cv2.cvtColor(cv2.imread(elon_test_path), cv2.COLOR_BGR2RGB)
    bill_gates_rgb = cv2.cvtColor(cv2.imread(bill_gates_path), cv2.COLOR_BGR2RGB)
    
    # Get face encodings
    elon_encodings = face_recognition.face_encodings(elon_image)  # type: ignore
    elon_test_encodings = face_recognition.face_encodings(elon_test)  # type: ignore
    bill_gates_encodings = face_recognition.face_encodings(bill_gates_image)  # type: ignore

    if not elon_encodings or not elon_test_encodings or not bill_gates_encodings:
        failed = [name for name, enc in [
            ("Elon musk.jpg", elon_encodings),
            ("Elon test.jpg", elon_test_encodings),
            ("Bill gates.jpg", bill_gates_encodings),
        ] if not enc]
        print(f"Error: could not detect a face in: {', '.join(failed)}")
        sys.exit(1)

    elon_encoding = elon_encodings[0]
    elon_test_encoding = elon_test_encodings[0]
    bill_gates_encoding = bill_gates_encodings[0]
    
    # Test 1: Compare two Elon images
    matches_elon = face_recognition.compare_faces([elon_encoding], elon_test_encoding)  # type: ignore
    distance_elon = face_recognition.face_distance([elon_encoding], elon_test_encoding)  # type: ignore
    
    print(f"When both images are Elon, you get a {matches_elon[0]} match with distance around ~{distance_elon[0]:.1f}.")
    
    # Test 2: Compare Elon with Bill Gates
    matches_bill = face_recognition.compare_faces([elon_encoding], bill_gates_encoding)  # type: ignore
    distance_bill = face_recognition.face_distance([elon_encoding], bill_gates_encoding)  # type: ignore
    
    print(f"When using Bill Gates as the test, you get {matches_bill[0]} and a larger distance (~{distance_bill[0]:.1f}).")