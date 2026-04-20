import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from attendance import AttendanceSystem
from utils import get_date


class FaceAttendanceApp:
    """Face recognition and attendance system without UI dependencies"""
    
    def __init__(self, enrollment_path='ImagesAttendance', attendance_path='data/Attendance'):
        self.enrollment_path = enrollment_path
        self.attendance_system = AttendanceSystem(attendance_path)
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_and_encode_faces()
    
    def load_and_encode_faces(self):
        """Load all enrolled faces and generate their encodings"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.enrollment_path):
            os.makedirs(self.enrollment_path)
            return
        
        for person_name in os.listdir(self.enrollment_path):
            person_path = os.path.join(self.enrollment_path, person_name)
            
            # Handle both single images and directories of images
            if os.path.isfile(person_path):
                # Single image file
                try:
                    img = face_recognition.load_image_file(person_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        # Use filename without extension as name
                        name = os.path.splitext(os.path.basename(person_path))[0]
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error encoding {person_path}: {e}")
            
            elif os.path.isdir(person_path):
                # Directory of images for one person
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        try:
                            img_path = os.path.join(person_path, img_file)
                            img = face_recognition.load_image_file(img_path)
                            encodings = face_recognition.face_encodings(img)
                            if encodings:
                                self.known_face_encodings.append(encodings[0])
                                self.known_face_names.append(person_name)
                        except Exception as e:
                            print(f"Error encoding {img_path}: {e}")
    
    def recognize_frame(self, frame, scale=0.25):
        """
        Recognize faces in a frame and mark attendance
        
        Args:
            frame: numpy array (BGR image from cv2)
            scale: scale factor for processing speed (default 0.25 for 4x speedup)
        
        Returns:
            annotated_frame: frame with rectangles and names drawn
            recognized_names: list of recognized names in this frame
        """
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), None, scale, scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and encode
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_names = []
        annotated_frame = frame.copy()
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.6
            )
            name = "Unknown"
            confidence = 0
            
            # Get distances for all known faces
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                # Only identify if confidence is high enough
                if matches[best_match_index] or face_distances[best_match_index] < 0.4:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    recognized_names.append(name)
                    # Mark attendance
                    self.attendance_system.mark_attendance(name)
            
            # Draw rectangle and label (scale back up for original frame)
            top, right, bottom, left = face_location
            top *= int(1 / scale)
            right *= int(1 / scale)
            bottom *= int(1 / scale)
            left *= int(1 / scale)
            
            # Color: green if recognized, red if unknown
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            thickness = 3
            
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, thickness)
            
            # Draw label background
            label_text = f"{name} ({confidence:.2f})" if name != "Unknown" else name
            cv2.rectangle(
                annotated_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            cv2.putText(
                annotated_frame, label_text, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return annotated_frame, recognized_names
    
    def enroll_person(self, name, image_path):
        """
        Enroll a new person from an image file
        
        Args:
            name: person's name
            image_path: path to the enrollment image
        
        Returns:
            success: boolean indicating if enrollment succeeded
        """
        try:
            person_dir = os.path.join(self.enrollment_path, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Load and validate image
            img = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(img)
            
            if len(face_locations) == 0:
                print(f"No face detected in {image_path}")
                return False
            
            if len(face_locations) > 1:
                print(f"Multiple faces detected in {image_path}. Using first face.")
            
            # Save the enrollment image
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            dest_path = os.path.join(person_dir, filename)
            
            # Use cv2 to save if image_path is a file path
            img_cv = cv2.imread(image_path)
            if img_cv is not None:
                cv2.imwrite(dest_path, img_cv)
            
            # Reload faces to update encodings
            self.load_and_encode_faces()
            return True
        
        except Exception as e:
            print(f"Error enrolling {name}: {e}")
            return False
    
    def enroll_from_array(self, name, image_array):
        """
        Enroll a new person from a numpy array (BGR format)
        
        Args:
            name: person's name
            image_array: numpy array in BGR format (from cv2 or webcam)
        
        Returns:
            success: boolean indicating if enrollment succeeded
        """
        try:
            person_dir = os.path.join(self.enrollment_path, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Convert BGR to RGB for face_recognition
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if len(face_locations) == 0:
                print(f"No face detected in image for {name}")
                return False
            
            if len(face_locations) > 1:
                print(f"Multiple faces detected. Using first face.")
            
            # Save the enrollment image
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            dest_path = os.path.join(person_dir, filename)
            cv2.imwrite(dest_path, image_array)
            
            # Reload faces to update encodings
            self.load_and_encode_faces()
            return True
        
        except Exception as e:
            print(f"Error enrolling {name}: {e}")
            return False
    
    def get_attendance_today(self):
        """
        Get today's attendance list
        
        Returns:
            list of dicts with Name, Time, Status
        """
        try:
            attendance_file = self.attendance_system.get_attendance_file()
            import pandas as pd
            df = pd.read_csv(attendance_file)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error reading attendance: {e}")
            return []
    
    def get_enrolled_persons(self):
        """
        Get list of all enrolled persons
        
        Returns:
            list of unique person names
        """
        return list(set(self.known_face_names))
