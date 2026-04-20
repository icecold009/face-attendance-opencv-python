import cv2
import numpy as np
import face_recognition
import os
import sys
from datetime import datetime

# Resolve ImagesAttendance relative to the project root (parent of src/)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

# b. Load images and class names
path = os.path.join(PROJECT_ROOT, 'ImagesAttendance')
images = []
classNames = []
_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
myList = [f for f in os.listdir(path) if f.lower().endswith(_IMAGE_EXTS)]
for cls in myList:
    curImg = cv2.imread(os.path.join(path, cls))
    if curImg is None:
        print(f"Warning: could not read image '{cls}', skipping.")
        continue
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


# c. Define encoding function
def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            print(f"Warning: no face found in enrollment image '{myList[idx]}', skipping.")
            continue
        encodeList.append(encodings[0])
    return encodeList


# d. Encode known faces
encodeListKnown = findEncodings(images)
print("Encoding Complete")


# e. Attendance CSV function
attendance_dir = os.path.join(PROJECT_ROOT, 'data', 'Attendance')
os.makedirs(attendance_dir, exist_ok=True)


def markAttendance(name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    attendance_file = os.path.join(attendance_dir, f'Attendance_{date_str}.csv')

    with open(attendance_file, 'a+', encoding='utf-8') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            if entry:
                nameList.append(entry[0])
        if name not in nameList:
            time_str = datetime.now().strftime('%H:%M:%S')
            f.write(f'{name},{time_str}\n')


# f. Initialize webcam
cap = cv2.VideoCapture(0)

# g. Real-time loop
while True:
    success, img = cap.read()
    if not success:
        print("Warning: failed to capture frame, retrying...")
        continue
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


