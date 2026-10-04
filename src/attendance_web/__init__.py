"""Flask blueprints for the attendance application."""

from flask import Flask

from .attendance import attendance_blueprint
from .dashboard import dashboard_blueprint
from .enrollment import enrollment_blueprint
from .recognition import recognition_blueprint


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(dashboard_blueprint)
    app.register_blueprint(recognition_blueprint)
    app.register_blueprint(enrollment_blueprint)
    app.register_blueprint(attendance_blueprint)


__all__ = [
    "attendance_blueprint",
    "dashboard_blueprint",
    "enrollment_blueprint",
    "recognition_blueprint",
    "register_blueprints",
]
