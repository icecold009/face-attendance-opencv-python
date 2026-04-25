# Contributing to Face Attendance System

Thank you for your interest in contributing! This guide will help you get set up quickly.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/<your-username>/face-attendance-opencv-python.git
cd face-attendance-opencv-python
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest flake8   # dev/test tools
```

### 4. Verify Your Setup

```bash
# Run the test suite
pytest tests/ -v

# Run the linter
flake8 src/ web_app.py --max-line-length=120 --select=E9,F63,F7,F82
```

---

## 🌿 Branching

- Branch from `main` using a descriptive name:
  - `feature/multi-camera-support`
  - `fix/attendance-duplicate-entry`
  - `docs/update-installation`

---

## ✅ Pull Request Checklist

Before opening a PR, please make sure:

- [ ] The existing tests still pass (`pytest tests/ -v`)
- [ ] New functionality includes tests where applicable
- [ ] Code follows the existing style (PEP 8, max line length 120)
- [ ] The README is updated if you changed behaviour or added features

---

## 🐛 Reporting Issues

Use the [GitHub issue tracker](https://github.com/icecold009/face-attendance-opencv-python/issues).  
Please include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behaviour
- Any relevant error output or screenshots

---

## 📄 License

By contributing you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
