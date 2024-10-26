import cv2
import numpy as np
import os
import csv
import pickle
import tkinter as tk
from tkinter import messagebox, Text, Scrollbar, Toplevel
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

class AttendanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Facial Recognition Attendance System")
        master.configure(bg="#f0f0f0")
        master.attributes("-fullscreen", True)  # Set to full screen

        # Title Label
        self.title_label = tk.Label(master, text="Facial Recognition Attendance", font=("Helvetica", 24), bg="#f0f0f0")
        self.title_label.pack(pady=10)

        # Name Entry Label
        self.label = tk.Label(master, text="Enter Your Name:", font=("Helvetica", 12), bg="#f0f0f0")
        self.label.pack(pady=5)

        # Entry Widget for name
        self.name_entry = tk.Entry(master, font=("Helvetica", 12), width=20)
        self.name_entry.pack(pady=5)

        # Start Attendance Button
        self.start_button = tk.Button(master, text="Start Attendance", command=self.start_attendance,
                                       bg="#4CAF50", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        self.start_button.pack(pady=10)

        # Add Person Button
        self.add_person_button = tk.Button(master, text="Add Person", command=self.open_add_person_window,
                                            bg="#2196F3", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        self.add_person_button.pack(pady=10)

        # Feedback Label
        self.feedback_label = tk.Label(master, text="", font=("Helvetica", 12), bg="#f0f0f0", fg="green")
        self.feedback_label.pack(pady=10)

        # Attendance History Button
        self.history_button = tk.Button(master, text="View Attendance History", command=self.view_attendance_history,
                                         bg="#FFC107", fg="black", font=("Helvetica", 12), padx=10, pady=5)
        self.history_button.pack(pady=5)

        # Quit Button
        self.quit_button = tk.Button(master, text="Quit", command=master.quit,
                                      bg="#f44336", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        self.quit_button.pack(pady=10)

        # Attendance List Display
        self.attendance_display = Text(master, height=10, width=50, font=("Helvetica", 12), state=tk.DISABLED)
        self.attendance_display.pack(pady=10)
        self.attendance_display.insert(tk.END, "Attendance Records:\n")

        # Scrollbar for Attendance List
        scrollbar = Scrollbar(master, command=self.attendance_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.attendance_display.config(yscrollcommand=scrollbar.set)

        # Initialize variables
        self.video = None
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.present_flags = {}
        self.attendance_active = False

    def open_add_person_window(self):
        self.add_person_window = Toplevel(self.master)
        self.add_person_window.title("Add Person")
        self.add_person_window.geometry("400x300")

        # Name Entry Label
        tk.Label(self.add_person_window, text="Enter Name:", font=("Helvetica", 12)).pack(pady=10)

        self.add_name_entry = tk.Entry(self.add_person_window, font=("Helvetica", 12), width=20)
        self.add_name_entry.pack(pady=10)

        # Counter Label
        self.counter_label = tk.Label(self.add_person_window, text="Faces Captured: 0/100", font=("Helvetica", 12))
        self.counter_label.pack(pady=10)

        # Capture Face Button
        capture_button = tk.Button(self.add_person_window, text="Capture Face", command=self.capture_face,
                                    bg="#4CAF50", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        capture_button.pack(pady=10)

        # Back Button
        back_button = tk.Button(self.add_person_window, text="Back", command=self.add_person_window.destroy,
                                bg="#f44336", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        back_button.pack(pady=10)

    def capture_face(self):
        name = self.add_name_entry.get()
        if not name:
            messagebox.showerror("Input Error", "Please enter a name.")
            return

        # Open video capture for face collection
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return

        facedata = []
        face_count = 0
        total_faces = 100

        while face_count < total_faces:  # Capture up to 100 samples
            ret, frame = self.video.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if face_count % 10 == 0:  # Collect every 10th frame
                    cropped_face = gray[y:y + h, x:x + w]
                    cropped_face = cv2.resize(cropped_face, (100, 100))
                    facedata.append(cropped_face.flatten())
                    face_count += 1

            self.counter_label.config(text=f"Faces Captured: {face_count}/{total_faces}")

            if face_count >= total_faces:
                break

        self.video.release()  # Release the video capture after use

        # Save face data and name
        self.save_face_data(name, facedata)

    def save_face_data(self, name, facedata):
        if not facedata:
            messagebox.showerror("Error", "No faces captured.")
            return

        # Check if data directory exists, if not create it
        if not os.path.exists('data'):
            os.makedirs('data')

        # Save face data
        face_data_file = 'data/face_data.pkl'
        if not os.path.isfile(face_data_file):
            with open(face_data_file, 'wb') as f:
                pickle.dump(np.array(facedata), f)
        else:
            with open(face_data_file, 'rb') as f:
                existing_data = pickle.load(f)
            updated_face_data = np.vstack([existing_data, np.array(facedata)])
            with open(face_data_file, 'wb') as f:
                pickle.dump(updated_face_data, f)

        # Save names
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

        messagebox.showinfo("Success", f"{name} added successfully!")

    def start_attendance(self):
        if self.attendance_active:
            messagebox.showinfo("Info", "Attendance session already active.")
            return

        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Input Error", "Please enter your name.")
            return

        # Load labels and face data
        try:
            with open('data/names.pkl', 'rb') as w:
                self.LABELS = pickle.load(w)
            with open('data/face_data.pkl', 'rb') as f:
                self.FACES = pickle.load(f)
        except Exception as e:
            messagebox.showerror("File Error", f"Error loading data: {e}")
            return

        # Setup KNN classifier
        try:
            self.knn = KNeighborsClassifier(n_neighbors=5)
            self.knn.fit(self.FACES, self.LABELS)
        except Exception as e:
            messagebox.showerror("Model Error", f"Error fitting model: {e}")
            return

        # Initialize present flags
        self.present_flags = {label: False for label in self.LABELS}
        self.attendance_active = True

        # Start video capture in background
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return

        self.master.after(100, self.update_frame)

    def update_frame(self):
        if not self.attendance_active:
            return

        ret, frame = self.video.read()
        if not ret:
            print("Failed to grab frame")
            self.stop_camera()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cropped_face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(cropped_face, (100, 100)).flatten().reshape(1, -1)
            prediction = self.knn.predict(resized_face)

            if prediction[0] == self.name_entry.get() and not self.present_flags[prediction[0]]:
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.record_attendance(prediction[0], timestamp)
                self.present_flags[prediction[0]] = True

                # Stop camera after successful attendance
                self.stop_camera()
                break  # Exit loop after marking present

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, prediction[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Continue updating frame
        self.master.after(100, self.update_frame)

    def record_attendance(self, name, timestamp):
        attendance_dir = 'Attendance'
        if not os.path.exists(attendance_dir):
            os.makedirs(attendance_dir)

        with open(f'{attendance_dir}/Attendance_{datetime.now().strftime("%m-%d-%Y")}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, timestamp])

        self.attendance_display.config(state=tk.NORMAL)
        self.attendance_display.insert(tk.END, f"{name} marked present at {timestamp}\n")
        self.attendance_display.config(state=tk.DISABLED)

    def stop_camera(self):
        if self.video is not None:
            self.video.release()
            self.video = None
            self.attendance_active = False

    def view_attendance_history(self):
        attendance_files = [f for f in os.listdir('Attendance') if f.endswith('.csv')]
        attendance_data = []

        for file in attendance_files:
            with open(os.path.join('Attendance', file), 'r') as f:
                reader = csv.reader(f)
                attendance_data.extend(list(reader))

        if not attendance_data:
            messagebox.showinfo("Attendance History", "No attendance records found.")
            return

        names = [row[0] for row in attendance_data]
        unique_names = set(names)
        attendance_counts = {name: names.count(name) for name in unique_names}

        # Show attendance counts as a bar chart
        import matplotlib.pyplot as plt
        plt.bar(attendance_counts.keys(), attendance_counts.values())
        plt.title("Attendance History")
        plt.xlabel("Names")
        plt.ylabel("Attendance Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def on_closing(self):
        self.stop_camera()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
