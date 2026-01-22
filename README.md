# Face Attendance System using OpenCV and Python

A Python-based face recognition system that automatically marks attendance using facial recognition. This system uses OpenCV for face detection and the `face_recognition` library for accurate face matching.

## Features

- **Face Enrollment**: Add new people to the system with multiple facial images
- **Real-time Face Recognition**: Detect and recognize faces from webcam feed
- **Automatic Attendance Marking**: Automatically records attendance when a recognized face is detected
- **Attendance Reports**: View today's attendance and historical records for individuals
- **User-friendly Interface**: Simple command-line menu for easy navigation

## Project Structure

```
face-attendance-opencv-python/
├── src/
│   ├── main.py                    # Main application entry point
│   ├── face_recognition_module.py # Face recognition logic
│   ├── attendance.py              # Attendance tracking and storage
│   └── utils.py                   # Utility functions
├── images/
│   └── Basic/                     # Stores training images for enrolled users
├── ImagesAttendance/              # Stores attendance verification images
├── data/
│   └── Attendance/                # Stores CSV files with attendance records
├── requirements.txt               # Python dependencies
└── encodings.pkl                  # Stores face encodings (generated at runtime)
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/icecold009/face-attendance-opencv-python.git
cd face-attendance-opencv-python
```

2. **Create a virtual environment** (optional but recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
cd src
python main.py
```

### Menu Options

1. **Enroll New Person**
   - Enter the person's name
   - Capture multiple images (press 'c' to capture, 'q' when done)
   - The system will automatically encode and store the face data
   - Press 'c' to capture at least 5-10 images for better accuracy

2. **Mark Attendance**
   - Starts the webcam feed for real-time face recognition
   - Recognized faces are highlighted in green with confidence score
   - Unknown faces are highlighted in red
   - Attendance is automatically marked (one entry per person per day)
   - Press 'q' to stop and view attendance summary

3. **View Today's Attendance**
   - Displays all attendance records for the current day
   - Shows Name, Time, and Status

4. **View Person History**
   - Search for a specific person's attendance history
   - Shows all attendance records across all days

5. **Exit**
   - Closes the application

## How It Works

### Face Enrollment Process
1. Captures multiple images of a person's face
2. Converts images to face encodings (128-dimensional vectors)
3. Stores encodings in `encodings.pkl` file
4. Associates each encoding with the person's name

### Attendance Marking Process
1. Reads frame from webcam
2. Detects all faces in the frame
3. Compares detected face encodings with stored encodings
4. If match confidence > 0.6, marks as recognized person
5. Records attendance with timestamp in daily CSV file

## Configuration

### Adjusting Recognition Tolerance

Edit `src/face_recognition_module.py`:
```python
self.face_recognition = FaceRecognitionSystem('encodings.pkl', tolerance=0.6)
```
- **Lower tolerance** (0.4-0.5): Stricter matching, fewer false positives
- **Higher tolerance** (0.6-0.7): More lenient, may have false positives

### Camera Selection

Edit `src/main.py`:
```python
cap = cv2.VideoCapture(0)  # 0 is default camera, use 1 or 2 for other cameras
```

## Requirements

- Python 3.7+
- OpenCV 4.8+
- face-recognition 1.3+
- numpy
- pandas
- Pillow
- openpyxl

## System Requirements

- **Webcam**: USB or built-in camera
- **Lighting**: Good lighting conditions for accurate face detection
- **Distance**: Keep face 30cm - 1m from camera
- **Processor**: Any modern CPU (Intel i3+, Ryzen 3+, or equivalent)
- **RAM**: 2GB minimum

## Tips for Best Results

1. **Enrollment Quality**
   - Enroll at least 5-10 images per person
   - Use different angles and lighting conditions
   - Ensure face is clearly visible and facing camera

2. **Recognition Accuracy**
   - Good lighting is crucial
   - Keep the camera steady
   - Maintain consistent distance from camera
   - Remove glasses or change appearance minimally between enrollment and recognition

3. **Performance**
   - First encoding might be slow
   - Subsequent runs will be faster
   - Use lower resolution for faster processing

## Troubleshooting

### "No module named 'face_recognition'"
```bash
pip install face-recognition
```

### "Failed to capture frame"
- Check if camera is connected
- Try different camera index (0, 1, 2)
- Ensure no other application is using the camera

### Low accuracy
- Enroll more images
- Use better lighting
- Ensure good face visibility
- Reduce tolerance value (more strict matching)

### Slow performance
- Reduce frame resolution
- Lower the number of stored encodings
- Use a more powerful computer

## Output Files

- **encodings.pkl**: Binary file storing all face encodings and names
- **data/Attendance/Attendance_YYYY-MM-DD.csv**: Daily attendance records with format:
  ```
  Name,Time,Status
  John,2024-01-22 09:30:15,Present
  Jane,2024-01-22 09:35:42,Present
  ```

## Limitations

- Works best with frontal face poses
- May struggle with extreme angles or partial faces
- Lighting conditions significantly affect accuracy
- Similar-looking individuals may cause false matches (adjust tolerance)

## Future Enhancements

- [ ] Web interface using Flask/Django
- [ ] Database backend (MySQL/PostgreSQL)
- [ ] Email/SMS notifications
- [ ] Report generation (PDF/Excel)
- [ ] Multi-face recognition in single frame
- [ ] Age and gender detection
- [ ] Liveness detection to prevent spoofing
- [ ] Mobile app integration

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.

---

**Created**: January 2026
**Last Updated**: January 22, 2026
