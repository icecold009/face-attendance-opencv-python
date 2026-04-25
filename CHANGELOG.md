# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2026-01-22

### Added
- Real-time face recognition and attendance marking via webcam
- Flask web dashboard with live video preview (5 fps)
- Face enrollment via webcam capture directly in the browser
- Daily attendance records exported to CSV (`data/Attendance/`)
- CLI interface (`src/main.py`) for server or terminal-only environments
- OpenCV-based face recognition engine (`src/face_recognition.py`) — runs 100% locally, no cloud APIs
- Support for per-person image directories under `ImagesAttendance/`
- `GET /attendance` API endpoint for today's attendance list
- `GET /enrolled-persons` API endpoint for enrolled person names
- `POST /recognize` API endpoint for single-frame recognition
- `POST /enroll` API endpoint for face enrollment
- `GET /health` health-check endpoint
- `--host` and `--port` CLI flags for `web_app.py`

---

## [Unreleased]

### Planned
- Database backend (PostgreSQL / MongoDB)
- Multi-camera support
- Liveness detection (anti-spoofing)
- REST API for external integration
- Docker containerisation
- Email / SMS notifications
- PDF / Excel report generation
