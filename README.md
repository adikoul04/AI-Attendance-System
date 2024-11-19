# AI Attendance System

## Overview
A modern, AI-powered web application for attendance tracking and dataset recording using facial recognition technology. It combines a Python backend with an interactive, responsive frontend interface.

## Features

- **Dataset Recording**:  
  - Captures and saves facial data via webcam using OpenCV.

- **Attendance Tracking**:  
  - Recognizes faces in real-time and logs attendance with timestamps.

- **Modern UI**:  
  - Blue-themed, responsive, with animations and dynamic feedback.

- **RESTful API**:  
  - Powered by Flask for seamless operations.

## Technologies Used

### Frontend
- **HTML5**: Structure of the web interface.
- **CSS3**: Modern blue canvas with gradient backgrounds and hover animations.
- **JavaScript (ES6)**: Handles user interactions and dynamic UI updates.
- **Google Fonts**: Poppins font for a clean and modern look.

### Backend
- **Flask**: Lightweight Python framework for building RESTful APIs.
- **OpenCV**: Facial recognition, image processing, and real-time face detection.
- **Subprocess**: Runs scripts for dataset recording and attendance tracking.
- **Threading**: Ensures non-blocking operations for asynchronous API calls.

### Storage
- **Pickle**: Serializes facial data for training and recognition.
- **CSV**: Logs attendance data with timestamps.

## How It Works

1. **Dataset Recording**:
   - Users enter their name in the frontend form and click **Start Recording**.
   - The backend triggers the `Dataset.py` script, which:
     - Activates the webcam.
     - Detects and captures facial data in real-time.
     - Saves the data for future use in face recognition.

2. **Attendance Tracking**:
   - Users click **Start Attendance** to activate face recognition.
   - The backend triggers the `Attendance.py` script, which:
     - Activates the webcam and scans for faces.
     - Matches detected faces against the pre-recorded dataset.
     - Logs recognized users' names and timestamps into a CSV file.

3. **Frontend Interaction**:
   - Provides real-time updates with dynamic messages and user-friendly controls.

## Use Cases

- **Educational Institutions**: Automates classroom attendance tracking.
- **Corporate Offices**: Tracks employee check-ins and check-outs.
- **Workshops & Conferences**: Monitors participant attendance seamlessly.

## Future Enhancements

- **Database Integration**: Store facial data and attendance logs in a robust database like PostgreSQL or MongoDB.
- **Enhanced Recognition Accuracy**: Use deep learning models like FaceNet for better face matching.
- **Mobile Compatibility**: Extend functionality to mobile platforms for greater accessibility.
- **Admin Dashboard**: Build a dashboard for visualizing attendance logs and managing datasets.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/adikoul04/AI4ALL-Computer-Vision-Project.git
   cd AI4ALL-Computer-Vision-Project


2. Set up Virtual Environment
   ```bash
   python -m venv face_rec_env
   source face_rec_env/bin/activate  # On Windows use: face_rec_env\Scripts\activate

   # Make sure to install necessary libraries in virtual environment if not installed
   # Ex: pip install scikit-learn

3. Start Flask Server
   ```bash
    python app.py
4. Open the application in your browser
   ```bash
   http://127.0.0.1:5000/
  
         

   
