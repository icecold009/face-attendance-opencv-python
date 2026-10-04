"""Dashboard and health routes."""

from datetime import datetime

from flask import Blueprint, jsonify, render_template


dashboard_blueprint = Blueprint("dashboard", __name__)


@dashboard_blueprint.get("/")
def index():
    return render_template("index.html")


@dashboard_blueprint.get("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})
