from flask import Flask, render_template, request, jsonify
from threading import Thread
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)
