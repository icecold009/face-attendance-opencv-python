# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2026-01-22

### Added
- Real-time face recognition and attendance marking via webcam
- Flask web interface with live video preview (5 fps)
- Daily attendance records exported to CSV (`data/Attendance/`)
- Alternate Flask server entrypoint (`src/main.py`) for server environments
- OpenCV-based fallback face recognition engine (`src/mock_face_recognition.py`) — runs 100% locally, no cloud APIs
- Support for per-person image directories under `ImagesAttendance/`
- `--host`, `--port`, and opt-in `--debug` flags for the Flask launchers

---

## [Unreleased]

### Added
- Dashboard API endpoints for recognition, enrollment, attendance, enrolled-person listing, and health checks

### Planned
- Database backend (PostgreSQL / MongoDB)
- Multi-camera support
- Liveness detection (anti-spoofing)
- REST API for external integration
- Docker containerisation
- Email / SMS notifications
- PDF / Excel report generation
