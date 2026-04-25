# 👤 Face Attendance System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()
[![CI](https://github.com/icecold009/face-attendance-opencv-python/actions/workflows/ci.yml/badge.svg)](https://github.com/icecold009/face-attendance-opencv-python/actions/workflows/ci.yml)

**A modern, local-first face recognition system for automatic attendance marking** 

[Features](#features) • [Quick Start](#-quick-start) • [Web UI](#-web-ui) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Features
- ✅ **Real-time Face Recognition** — ~5 fps live detection
- ✅ **Auto Attendance Marking** — One entry per person per day
- ✅ **Face Enrollment** — Add new people with simple UI
- ✅ **Attendance Reports** — View daily records and history
- ✅ **Zero Dependencies** — Runs 100% locally, no cloud

</td>
<td width="50%">

### 🚀 Advanced Features  
- 🌐 **Modern Web UI** — Local Flask dashboard with live webcam
- 📊 **Statistics** — Track enrolled people & daily attendance
- 🎨 **Real-time Annotation** — Green/red boxes for faces
- 📁 **CSV Export** — Automatic daily attendance records
- ⚡ **Fast Processing** — Optimized for low-end systems

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Two Modes of Operation

#### **Mode 1: Web UI (Recommended)** 🌐
Perfect for modern browsers, real-time visual feedback, and team use.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web server
python web_app.py

# 3. Open browser
# Navigate to http://localhost:5000
```

#### **Mode 2: Command Line** 💻
Traditional CLI interface for server environments.

```bash
# Run the main program
cd src
python main.py
```

---

## 🌐 Web UI

### Features
- 📹 **Live Video Preview** — Real-time webcam feed in browser
- 🎬 **Live Recognition** — 5 fps frame sampling and annotation
- 📝 **Enroll People** — Add faces directly from webcam
- 📊 **View Attendance** — See today's records in real-time
- 📊 **Statistics Dashboard** — Count of enrolled people and attendance

### Access
- **URL**: `http://localhost:5000`
- **Protocol**: HTTP only (localhost)
- **Browser Support**: Chrome, Firefox, Edge, Safari
- **Cost**: $0 (runs entirely on your machine)

### Screenshots

| Screen | Description |
|--------|-------------|
| **Dashboard** | Start/Stop recognition, enroll new people, view live attendance stats |
| **Live Feed** | Side-by-side: raw webcam video + annotated recognition results |
| **Detection** | Green bounding boxes for recognised faces, red for unknown visitors |
| **Enrollment** | Step-by-step: capture → enter name → submit → encoding saved automatically |

> 💡 Run the app locally and capture your own screenshots to add here!

---

## 📋 Project Structure

```
face-attendance-opencv-python/
│
├── web_app.py                      # Flask server (entry point)
├── templates/
│   └── index.html                  # Web UI (vanilla JS)
│
├── src/
│   ├── face_attendance_app.py       # Core recognition engine (web)
│   ├── basic_face_recognition.py   # Alternate recognition engine (CLI)
│   ├── attendance.py               # Attendance tracking
│   ├── face_recognition.py         # Local face encoding & matching (no dlib)
│   ├── utils.py                    # Utility functions
│   └── main.py                     # CLI entry point
│
├── ImagesAttendance/               # Enrollment images
├── images/
│   └── Basic/                      # Demo images for CLI mode
├── data/
│   └── Attendance/                 # Daily CSV records (auto-created)
│
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## 💾 Installation

### Prerequisites
- **Python**: 3.7 or higher
- **Webcam**: USB or built-in camera
- **OS**: Windows, macOS, Linux

### Step 1: Clone Repository
```bash
git clone https://github.com/icecold009/face-attendance-opencv-python.git
cd face-attendance-opencv-python
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Guide

### Web UI Workflow

#### 1️⃣ Start Recognition
```
Click "Start Recognition" → Live video appears
→ Face detection runs at 5 fps
→ Green boxes for matches, red for unknown
→ Attendance auto-marks for recognized faces
→ Click "Stop Recognition" to pause
```

#### 2️⃣ Enroll New Person
```
Click "Enroll New Person" → Enter name
→ Click "Capture Face for Enrollment"
→ Position face in camera for 1-2 seconds
→ Click "Submit Enrollment"
→ Face encoding is saved automatically
```

#### 3️⃣ View Attendance
```
Click "View Attendance" → See today's records
→ Shows Name, Time, Status
→ Automatically updates in real-time
```

### CLI Workflow

```bash
cd src
python main.py

# Menu options:
# 1. Enroll New Person
# 2. Mark Attendance
# 3. View Today's Attendance
# 4. View Person History
# 5. Exit
```

---

## ⚙️ Configuration

### Adjust Recognition Tolerance
Edit `src/face_attendance_app.py`:
```python
tolerance=0.6  # 0.4 (strict) to 0.7 (lenient)
```

### Change Camera
Edit `web_app.py` or `src/main.py`:
```python
cap = cv2.VideoCapture(0)  # 0=default, 1=USB, 2=external
```

### Custom Frame Rate
Edit `templates/index.html`:
```javascript
const FPS = 5;  // Frames per second (adjust for speed/accuracy)
```

---

## 📊 How It Works

### Face Recognition Pipeline
```
Frame Input
    ↓
Detect Faces (Haar Cascade)
    ↓
Extract Face Regions
    ↓
Generate 128-D Encodings (HOG-based gradient features)
    ↓
Compare with Known Encodings
    ↓
Match? (distance < 0.6)
    ├─ YES → Mark Attendance + Green Box
    └─ NO  → Red Box (Unknown)
```

### Attendance Storage
```
Daily File: data/Attendance/Attendance_2024-01-22.csv
Format:
    Name,Time,Status
    John,09:30:15,Present
    Jane,09:35:42,Present
```

---

## 📈 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| **CPU** | Intel i3 / Ryzen 3 | Intel i7 / Ryzen 5+ |
| **RAM** | 2 GB | 4 GB+ |
| **Storage** | 100 MB | 500 MB+ |
| **Python** | 3.7 | 3.9+ |
| **OS** | Win/Mac/Linux | Win/Mac/Linux |

---

## 🎯 Tips for Best Results

### ✅ For Enrollment
- Capture **5-10 images** per person
- Use **different angles** and lighting
- Ensure **face is clearly visible**
- Good lighting (face 30cm-1m from camera)

### ✅ For Recognition
- **Consistent lighting** is crucial
- **Steady camera** position
- **No glasses** or minimal appearance changes
- **Frontal face** angles work best

### ⚡ For Performance
- Lower tolerance (0.4) = Faster but stricter
- Reduce FPS if system is slow
- Use powerful GPU if available

---

## 🐛 Troubleshooting

### "No module named 'face_recognition'"
The project ships its own local `src/face_recognition.py` module (no installation needed).
If you see this error, make sure you are running the app from the repository root so that
`src/` is on the Python path, or run via the provided entry points:
```bash
python web_app.py        # web UI
cd src && python main.py # CLI
```

### Webcam not detected
```bash
# Try different camera index
# In code, change: cv2.VideoCapture(0) → cv2.VideoCapture(1)
```

### Low recognition accuracy
- ✅ Enroll more images (10+)
- ✅ Check lighting conditions
- ✅ Reduce tolerance to 0.4-0.5

### Slow performance
- 🔧 Reduce FPS from 5 to 2
- 🔧 Close other applications
- 🔧 Use better processor

---

## 🌐 Sharing Your Project

### Local Network Access
```bash
# Run with --host 0.0.0.0
python web_app.py --host 0.0.0.0
# Access from: http://YOUR_IP:5000
```

### Remote Access
Use **ngrok** for quick sharing (free):
```bash
ngrok http 5000
# Get public URL: https://abc123.ngrok.io
```

### Deploy Online
- **Render**: Free tier available
- **Railway**: Supports Python
- **Heroku**: Traditional choice

---

## 📦 Dependencies

```
opencv-python==4.8.1.78       # Computer vision
numpy==1.24.3                 # Numerical computing
flask>=2.0.0                  # Web server
pandas>=2.0.3                 # Data handling
Pillow>=10.0.0                # Image processing
```

> **No heavy external dependencies**: face recognition runs via a built-in local module
> (`src/face_recognition.py`) using OpenCV Haar cascades + HOG features — no dlib or cloud APIs required.

---

## 📄 Output Files

### Enrollment Data
```
ImagesAttendance/
├── John/
│   ├── John_20240122_093015.jpg
│   └── John_20240122_093045.jpg
└── Jane/
    └── Jane_20240122_093115.jpg
```

### Attendance Records
```
data/Attendance/Attendance_2024-01-22.csv
Name,Time,Status
John,09:30:15,Present
Jane,09:35:42,Present
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🚀 Future Enhancements

- [ ] Database backend (PostgreSQL/MongoDB)
- [ ] Multi-camera support
- [ ] Face mask detection
- [ ] Liveness detection (prevent spoofing)
- [ ] REST API for external integration
- [ ] Docker containerization
- [ ] Mobile app companion
- [ ] Real-time notifications
- [ ] Email/SMS notifications
- [ ] Report generation (PDF/Excel)
- [ ] Age and gender detection

---

<div align="center">

**Made with ❤️ by [icecold009](https://github.com/icecold009)**

⭐ If you found this helpful, please consider starring the repository!

</div>
