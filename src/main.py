import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basic_face_recognition import FaceRecognitionSystem
from attendance import AttendanceSystem
from utils import create_directory, resize_image, get_date

class FaceAttendanceApp:
    def __init__(self):
        self.face_recognition = FaceRecognitionSystem('encodings.pkl')
        self.attendance = AttendanceSystem('data/Attendance')
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        self.running = True
        
        # Create necessary directories
        create_directory('images/Basic')
        create_directory('ImagesAttendance')
        create_directory('data/Attendance')
    
    def enroll_new_person(self):
        """Enroll a new person"""
        print("\n--- Enroll New Person ---")
        name = input("Enter person's name: ").strip()
        
        if not name:
            print("Name cannot be empty!")
            return
        
        person_path = f'images/Basic/{name}'
        create_directory(person_path)
        
        print(f"Capturing images for {name}. Press 'c' to capture, 'q' to quit...")
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = resize_image(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Captured: {count}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(f"Enrolling {name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    cv2.imwrite(f'{person_path}/{count}.jpg', face_img)
                    count += 1
                    print(f"Image {count} captured")
                else:
                    print("No face detected!")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Encode faces
        print(f"Encoding {count} images for {name}...")
        added = self.face_recognition.add_multiple_images(name, person_path)
        
        if added:
            print(f"Successfully enrolled {name}!")
        else:
            print(f"Failed to encode images for {name}")
    
    def start_attendance(self):
        """Start real-time attendance marking"""
        print("\n--- Starting Attendance System ---")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        self.attendance.reset_daily_marked()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = resize_image(frame, width=640)
            
            # Recognize faces
            face_locations, face_names = self.face_recognition.recognize_faces(frame)
            
            # Draw faces
            frame = self.face_recognition.draw_faces(frame, face_locations, face_names)
            
            # Mark attendance
            for name, confidence in face_names:
                if confidence > 0.6:
                    self.attendance.mark_attendance(name)
            
            # Display info
            date_info = f"Date: {get_date()}"
            cv2.putText(frame, date_info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            marked_info = f"Marked: {len(self.attendance.marked_today)}"
            cv2.putText(frame, marked_info, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Attendance System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        summary = self.attendance.get_attendance_summary()
        if summary is not None and not summary.empty:
            print("\n--- Attendance Summary ---")
            print(summary.to_string(index=False))
    
    def view_attendance(self):
        """View today's attendance"""
        print("\n--- Today's Attendance ---")
        summary = self.attendance.get_attendance_summary()
        
        if summary is not None and not summary.empty:
            print(summary.to_string(index=False))
        else:
            print("No attendance records for today")
    
    def view_person_history(self):
        """View attendance history for a person"""
        print("\n--- View Person History ---")
        name = input("Enter person's name: ").strip()
        
        history = self.attendance.get_person_attendance_history(name)
        
        if history is not None and not history.empty:
            print(f"\nAttendance history for {name}:")
            print(history.to_string(index=False))
        else:
            print(f"No attendance records found for {name}")
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*40)
        print("  Face Attendance System")
        print("="*40)
        print("1. Enroll New Person")
        print("2. Mark Attendance")
        print("3. View Today's Attendance")
        print("4. View Person History")
        print("5. Exit")
        print("="*40)
    
    def run(self):
        """Main application loop"""
        while self.running:
            self.show_menu()
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                self.enroll_new_person()
            elif choice == '2':
                self.start_attendance()
            elif choice == '3':
                self.view_attendance()
            elif choice == '4':
                self.view_person_history()
            elif choice == '5':
                print("Exiting...")
                self.running = False
            else:
                print("Invalid choice! Please try again.")

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.run()
