"""
CI-safe entrypoint wrapper for the face attendance application.

- Exposes `app` for tests: `from src.main import app`
- Only starts the server when executed directly: `python -m src.main`
"""

from .face_attendance_app import create_app  # noqa: F401

# Create the Flask app instance for tests or external use
app = create_app()

if __name__ == "__main__":
    # Only run the dev server when executed directly
    app.run(host="0.0.0.0", port=5000, debug=True)