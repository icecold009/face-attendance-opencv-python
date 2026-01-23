import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# b. Load images and class names
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(os.path.join(path, cls))
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


# c. Define encoding function
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# d. Encode known faces
encodeListKnown = findEncodings(images)
print("Encoding Complete")


# e. Attendance CSV function
attendance_dir = os.path.join('data', 'Attendance')
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


