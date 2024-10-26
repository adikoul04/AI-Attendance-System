import cv2
import numpy as np
import os
import pickle

# Open the default camera
video = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

facedata = []
i = 0
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Detected {len(faces)} faces")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if len(facedata) < 100 and i % 10 == 0:
            cropped_face = gray[y:y+h, x:x+w]
            cropped_face = cv2.resize(cropped_face, (100, 100))  # Resize face so all images have the same size
            facedata.append(cropped_face.flatten())
            print("Collected faces count: ", len(facedata))

    cv2.imshow('frame', frame)
    i += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print("Total faces collected: ", len(facedata))
video.release()
cv2.destroyAllWindows()

# Convert list to numpy array for better handling
face_data = np.array(facedata)

# Check if data directory exists, if not create it
if not os.path.exists('data'):
    os.makedirs('data')

# Save or append face data
face_data_file = 'data/face_data.pkl'
if not os.path.isfile(face_data_file):
    with open(face_data_file, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(face_data_file, 'rb') as f:
        existing_data = pickle.load(f)
    updated_face_data = np.vstack([existing_data, face_data])
    with open(face_data_file, 'wb') as f:
        pickle.dump(updated_face_data, f)

# Save or append names
names_file = 'data/names.pkl'
names_data = [name] * len(facedata)
if not os.path.isfile(names_file):
    with open(names_file, 'wb') as f:
        pickle.dump(names_data, f)
else:
    with open(names_file, 'rb') as f:
        existing_names = pickle.load(f)
    updated_names = existing_names + names_data
    with open(names_file, 'wb') as f:
        pickle.dump(updated_names, f)
