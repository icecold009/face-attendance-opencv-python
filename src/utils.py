import cv2
import numpy as np
import os
from datetime import datetime
import pickle

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio"""
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def get_faces(image, detector):
    """Detect faces in an image using cascade classifier"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    return faces

def draw_rectangle(image, faces):
    """Draw rectangles around detected faces"""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_pickle(data, filepath):
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """Load data from pickle file"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_date():
    """Get current date as string"""
    return datetime.now().strftime("%Y-%m-%d")
