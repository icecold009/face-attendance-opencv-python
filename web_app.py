from __future__ import annotations

import argparse
import os
import sys

# Add src directory to path to import the application factory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from face_attendance_app import create_app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Attendance Web App")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind (default: 5000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("Starting Face Attendance Web App...")
    print(f"Open http://{args.host}:{args.port} in your browser")
    app.run(debug=args.debug, host=args.host, port=args.port)
