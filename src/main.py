"""
Entry point wrapper for the face attendance application.

This keeps backward compatibility with tests and scripts
that expect `python -m src.main` or `python src/main.py`.
"""

from .face_attendance_app import app  # noqa: F401

if __name__ == "__main__":
    # Optional: only run the Flask server if executed directly.
    # In tests, they can still import `app` without side effects.
    app.run(host="0.0.0.0", port=5000, debug=True)