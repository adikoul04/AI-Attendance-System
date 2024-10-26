import cv2
import numpy as np
import os
import csv
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Check if Attendance directory exists, if not create it
attendance_dir = 'Attendance'
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load labels and face data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Setup KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Prepare to collect attendance data
COL_NAMES = ['NAME', 'TIME']
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cropped_face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (100, 100)).flatten().reshape(1, -1)
        prediction = knn.predict(resized_face)
        timestamp = datetime.now().strftime('%H:%M:%S')

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, prediction[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save or update the attendance record
        with open(f'{attendance_dir}/Attendance_{datetime.now().strftime("%m-%d-%Y")}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([prediction[0], timestamp])

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()










