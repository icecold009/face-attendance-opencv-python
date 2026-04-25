from __future__ import annotations

import argparse
import os
import sys
import base64
import cv2
import numpy as np
from io import BytesIO
from typing import Any
from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime

# Add src directory to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from face_attendance_app import FaceAttendanceApp

app = Flask(__name__)

# Initialize the attendance app
attendance_app = FaceAttendanceApp(
    enrollment_path='ImagesAttendance',
    attendance_path='data/Attendance'
)


@app.route('/')
def index() -> str:
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize() -> Response:
    """
    Handle face recognition on a frame.
    Expects:
        - frame: base64-encoded JPEG image
    Returns:
        - annotated_frame: base64-encoded JPEG with rectangles/labels
        - recognized_names: list of names detected in this frame
        - timestamp: server timestamp
    """
    try:
        data = request.get_json()
        frame_b64 = data.get('frame')
        
        if not frame_b64:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode base64 to image
        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Run recognition
        annotated_frame, recognized_names = attendance_app.recognize_frame(frame)
        
        # Encode result back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'annotated_frame': result_b64,
            'recognized_names': recognized_names,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/enroll', methods=['POST'])
def enroll() -> Response:
    """
    Enroll a new person.
    Expects:
        - name: person's name
        - frame: base64-encoded JPEG image
    Returns:
        - success: boolean
        - message: status message
    """
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        frame_b64 = data.get('frame')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        if not frame_b64:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode base64 to image
        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Enroll the person
        success = attendance_app.enroll_from_array(name, frame)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully enrolled {name}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to enroll {name}. No face detected or error occurred.'
            }), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/attendance', methods=['GET'])
def get_attendance() -> Response:
    """
    Get today's attendance list.
    Returns:
        - attendance: list of dicts with Name, Time, Status
        - date: today's date
    """
    try:
        attendance_list = attendance_app.get_attendance_today()
        return jsonify({
            'success': True,
            'attendance': attendance_list,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'count': len(attendance_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/enrolled-persons', methods=['GET'])
def get_enrolled_persons() -> Response:
    """
    Get list of all enrolled persons.
    Returns:
        - persons: list of person names
    """
    try:
        persons = attendance_app.get_enrolled_persons()
        return jsonify({
            'success': True,
            'persons': persons,
            'count': len(persons)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health() -> Response:
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Attendance Web App')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print("Starting Face Attendance Web App...")
    print(f"Open http://{args.host}:{args.port} in your browser")
    app.run(debug=args.debug, host=args.host, port=args.port)
