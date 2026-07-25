"""CI-safe entrypoint wrapper for the face attendance application."""

import argparse
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
    parser = argparse.ArgumentParser(description="Face Attendance App")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind (default: 5000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    app.run(debug=args.debug, host=args.host, port=args.port)
