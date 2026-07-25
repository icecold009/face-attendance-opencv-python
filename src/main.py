"""CI-safe entrypoint wrapper for the face attendance application."""

import sys
from pathlib import Path


src_dir = Path(__file__).resolve().parent
repo_root = src_dir.parent
for import_path in (src_dir, repo_root):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


if __package__:
    from .face_attendance_app import create_app
else:
    # Support the README's standalone invocation: ``cd src && python main.py``.
    from face_attendance_app import create_app

# Create the Flask app instance for tests or external use
app = create_app()

if __name__ == "__main__":
    # Only run the dev server when executed directly
    app.run(host="0.0.0.0", port=5000, debug=True)
