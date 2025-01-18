from flask import Flask, render_template, request, jsonify, send_from_directory
from threading import Thread
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/attendance')
def attendance_viewer():
    return render_template('attendance_viewer.html')

@app.route('/start_dataset', methods=['POST'])
def start_dataset():
    data = request.get_json()
    name = data.get('name', 'Unknown')
    Thread(target=lambda: subprocess.run(['python', 'Dataset.py'], input=name.encode())).start()
    return jsonify({"status": "Dataset recording started", "name": name})

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    Thread(target=lambda: subprocess.run(['python', 'Attendance.py'])).start()
    return jsonify({"status": "Attendance tracking started"})

@app.route('/Attendance/<path:filename>')
def serve_attendance_file(filename):
    return send_from_directory('Attendance', filename)

if __name__ == '__main__':
    app.run(debug=True)