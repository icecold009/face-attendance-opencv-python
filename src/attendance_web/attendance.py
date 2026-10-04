"""Attendance summary and enrolled-person routes."""

from datetime import datetime

from flask import Blueprint, current_app, jsonify


attendance_blueprint = Blueprint("attendance", __name__)


@attendance_blueprint.get("/attendance")
def get_attendance():
    summary = current_app.config["ATTENDANCE_SYSTEM"].get_attendance_summary()
    attendance = [] if summary is None else summary.to_dict(orient="records")
    return jsonify(
        {
            "success": True,
            "attendance": attendance,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "count": len(attendance),
        }
    )


@attendance_blueprint.get("/enrolled-persons")
def get_enrolled_persons():
    persons = current_app.config["IMAGE_STORE"].list_people()
    return jsonify({"success": True, "persons": persons, "count": len(persons)})
