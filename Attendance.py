import cv2
import numpy as np
import os
import csv
import pickle
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
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Track attendance for the day
attendance_date = datetime.now().strftime("%m-%d-%Y")
attendance_file = f'{attendance_dir}/Attendance_{attendance_date}.csv'
daily_attendance = set()  # Keep track of who's already been marked present

# Create attendance file with headers if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Time'])

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
        predicted_name = prediction[0]
        
        # Only record attendance if this person hasn't been marked present today
        if predicted_name not in daily_attendance:
            timestamp = datetime.now().strftime('%H:%M:%S')
            daily_attendance.add(predicted_name)
            
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([predicted_name, timestamp])
            
            print(f"Marked attendance for {predicted_name}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()