# student_management_opencv_custom.py
from flask import Flask, render_template_string, request, redirect, url_for, session, flash, jsonify, Response
import hashlib
import datetime
import sqlite3
import cv2
import numpy as np
import pickle
import base64
import os
import time
import warnings
import threading
from contextlib import contextmanager

# Suppress the semaphore warning
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

app = Flask(__name__)
app.secret_key = 'student-management-secret-key-2024-opencv-custom'
  
# --- Global variables for webcam and attendance state ---
video_capture = None
capture_lock = threading.Lock()
attendance_log_lock = threading.Lock()
marked_today_set = set() # A set to hold IDs marked today to prevent re-marking

# Base HTML Template (Unchanged)
BASE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --success: #4cc9f0;
            --dark: #1d3557;
            --light: #f8f9fa;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
        }

        .camera-feed {
            background: #1a1a1a;
            border-radius: 16px;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            position: relative;
            overflow: hidden;
        }

        /* Style for the video stream image */
        #video-stream {
           width: 100%;
    height: auto;
    border-radius: 16px;
    object-fit: cover;
        }

        .face-overlay {
            position: absolute;
            border: 3px solid #4cc9f0;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(76, 201, 240, 0.5);
        }

        .recognition-active {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(76, 201, 240, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(76, 201, 240, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 201, 240, 0); }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);">
        <div class="container">
            <a class="navbar-brand fw-bold text-dark" href="/">
                <i class="fas fa-graduation-cap me-2"></i>Facial Attendance System
            </a>
            {% if session.user_id %}
            <div class="navbar-nav ms-auto">
                <div class="nav-item dropdown">
                    <a class="nav-link dropdown-toggle text-dark" href="#" data-bs-toggle="dropdown">
                        <i class="fas fa-user-circle me-1"></i>
                        {% if session.role == 'student' %}
                            {{ session.get('student_name', 'Student') }}
                        {% else %}
                            {{ session.user_id }} {% endif %}
                    </a>
                    <ul class="dropdown-menu">
                        <li><a class="dropdown-item" href="/logout"><i class="fas fa-sign-out-alt me-2"></i>Logout</a></li>
                    </ul>
                </div>
            </div>
            {% endif %}
        </div>
    </nav>

    <div class="container mt-4">
        {% for category, message in get_flashed_messages(with_categories=true) %}
            <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} alert-dismissible fade show glass-card">
                {{ message }}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        {% endfor %}
    </div>

    <main class="container my-4">
        {{ content | safe }}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# DatabaseManager Class (Unchanged)
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('students_opencv_custom.db', check_same_thread=False) # Use a new DB file
        self.conn.execute("PRAGMA foreign_keys = ON") # Enable foreign key constraints
        self.create_tables()

    @contextmanager
    def get_cursor(self):
        """Context manager for database operations"""
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()

    def create_tables(self):
        with self.get_cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admins(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS teachers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            ''')

            # face_encoding will store a pickled face descriptor (LBPH histogram)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students(
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    year TEXT NOT NULL,
                    contact TEXT NOT NULL,
                    face_encoding BLOB
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT DEFAULT 'Present',
                    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE
                )
            ''')

            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date)')

            # Insert sample data
            try:
                admin_pass = hashlib.sha256("admin123".encode()).hexdigest()
                cursor.execute('INSERT OR IGNORE INTO admins (username, password) VALUES (?, ?)',
                               ('admin', admin_pass))

                hashed_password = hashlib.sha256("pass123".encode()).hexdigest()
                cursor.execute('INSERT OR IGNORE INTO teachers (username, password) VALUES (?, ?)',
                               ('teacher1', hashed_password))

                sample_students = [
                    ('S1001', 'John Doe', 'Computer Science', '2023', '1234567890'),
                    ('S1002', 'Jane Smith', 'Electrical Engineering', '2023', '9876543210'),
                    ('S1003', 'Mike Johnson', 'Mechanical Engineering', '2024', '5551234567'),
                    ('S1004', 'Sarah Wilson', 'Computer Science', '2024', '4445556666'),
                    ('S1005', 'David Brown', 'Electrical Engineering', '2023', '7778889999')
                ]
                for student in sample_students:
                    cursor.execute('INSERT OR IGNORE INTO students (id, name, department, year, contact, face_encoding) VALUES (?, ?, ?, ?, ?, NULL)', student)
                    # Clear old encodings just in case
                    cursor.execute('UPDATE students SET face_encoding = NULL WHERE id = ?', (student[0],))

            except Exception as e:
                print(f"Sample data warning: {e}")

db = DatabaseManager()

# --- AdvancedFaceRecognition Class (CUSTOM IMPLEMENTATION using OpenCV only) ---
class AdvancedFaceRecognition:
    def __init__(self):
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Face recognition - Using LBPH (Local Binary Patterns Histograms)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for face data
        self.face_descriptors = []  # List of face descriptors (histograms)
        self.face_labels = []       # List of corresponding student IDs
        self.label_to_id = {}       # Mapping from label index to student_id
        self.id_to_name = {}        # Mapping from student_id to name
        
        # Recognition parameters
        self.recognition_buffer = {}
        self.RECOGNITION_THRESHOLD = 5
        self.CONFIDENCE_THRESHOLD = 70  # Lower is better match for LBPH
        self.RESIZE_FACTOR = 0.5
        self.is_trained = False
        
        self.load_trained_data()

    def load_trained_data(self):
        """Load trained face data from the database."""
        print("ℹ️  Loading face data...")
        self.face_descriptors = []
        self.face_labels = []
        self.label_to_id = {}
        self.id_to_name = {}
        
        try:
            with db.get_cursor() as cursor:
                students = cursor.execute(
                    "SELECT id, name, face_encoding FROM students WHERE face_encoding IS NOT NULL"
                ).fetchall()

            labels_count = 0
            for student_id, student_name, face_encoding_blob in students:
                try:
                    # Each student's encoding contains multiple face samples
                    face_samples = pickle.loads(face_encoding_blob)
                    
                    if isinstance(face_samples, list) and len(face_samples) > 0:
                        for sample in face_samples:
                            if isinstance(sample, np.ndarray):
                                self.face_descriptors.append(sample)
                                self.face_labels.append(labels_count)
                        
                        self.label_to_id[labels_count] = student_id
                        self.id_to_name[student_id] = student_name
                        labels_count += 1
                        
                except Exception as e:
                    print(f"⚠️  Error loading encoding for {student_id}: {e}")

            if len(self.face_descriptors) > 0:
                # Train the recognizer
                try:
                    self.face_recognizer.train(self.face_descriptors, np.array(self.face_labels))
                    self.is_trained = True
                    print(f"✅ Loaded {labels_count} students with {len(self.face_descriptors)} total face samples.")
                except Exception as e:
                    print(f"❌ Error training recognizer: {e}")
                    self.is_trained = False
            else:
                self.is_trained = False
                print("ℹ️  No trained face data found.")
                
        except Exception as e:
            print(f"❌ Error loading face data from DB: {e}")
            self.is_trained = False

    def preprocess_face(self, face_image):
        """Preprocess face image for better recognition."""
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
            
        # Resize to standard size
        standard_size = (100, 100)
        resized = cv2.resize(gray, standard_size)
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(resized)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return blurred

    def extract_face_descriptor(self, face_image):
        """Extract LBPH descriptor from face image."""
        preprocessed = self.preprocess_face(face_image)
        return preprocessed

    def capture_and_save_face_encoding(self, student_id):
        """Generator: Captures multiple face samples and saves them."""
        global video_capture
        captured_samples = []
        
        with capture_lock:
            if video_capture is None:
                video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                print("❌ Cannot open webcam for capture")
                video_capture = None
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(err_frame, "Error: Cannot open webcam", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                ret_enc, buffer = cv2.imencode('.jpg', err_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return

        print(f"ℹ️  Capturing face samples for {student_id}...")

        try:
            capture_attempts = 0
            max_attempts = 100  # Try for ~10 seconds
            samples_needed = 10  # Capture multiple samples for better recognition
            samples_captured = 0

            while capture_attempts < max_attempts and samples_captured < samples_needed:
                ret, frame = video_capture.read()
                if not ret:
                    time.sleep(0.1)
                    capture_attempts += 1
                    continue

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=self.RESIZE_FACTOR, fy=self.RESIZE_FACTOR)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                status_text = "Look at camera - No face"
                box_color = (0, 0, 255)  # Red

                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    status_text = f"Face detected! Capturing... ({samples_captured+1}/{samples_needed})"
                    box_color = (0, 255, 0)  # Green

                    # Scale back up coordinates
                    scale = 1 / self.RESIZE_FACTOR
                    x = int(x * scale)
                    y = int(y * scale)
                    w = int(w * scale)
                    h = int(h * scale)

                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Extract descriptor and store
                    face_descriptor = self.extract_face_descriptor(face_roi)
                    captured_samples.append(face_descriptor)
                    samples_captured += 1

                    # Draw success indicator
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "CAPTURED!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                elif len(faces) > 1:
                    status_text = "Multiple faces detected!"
                    box_color = (0, 165, 255)  # Orange

                # Draw bounding boxes and status
                for (x, y, w, h) in faces:
                    scale = 1 / self.RESIZE_FACTOR
                    x = int(x * scale)
                    y = int(y * scale)
                    w = int(w * scale)
                    h = int(h * scale)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                cv2.putText(frame, f"Samples: {samples_captured}/{samples_needed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Yield frame
                ret_enc, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                capture_attempts += 1
                time.sleep(0.2)  # Small delay between captures

            # Save to database if we got enough samples
            if len(captured_samples) >= 5:  # Minimum 5 samples required
                encoding_blob = pickle.dumps(captured_samples)
                with db.get_cursor() as cursor:
                    cursor.execute(
                        "UPDATE students SET face_encoding = ? WHERE id = ?",
                        (encoding_blob, student_id)
                    )
                print(f"✅ Saved {len(captured_samples)} face samples for {student_id}")
                
                # Show success message
                success_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(success_frame, "SUCCESS!", (240, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(success_frame, f"Captured {len(captured_samples)} samples", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret_enc, buffer = cv2.imencode('.jpg', success_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(2)
            else:
                print(f"⚠️  Insufficient samples captured for {student_id}")
                fail_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(fail_frame, "INSUFFICIENT SAMPLES", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(fail_frame, f"Only {len(captured_samples)} captured (need at least 5)", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret_enc, buffer = cv2.imencode('.jpg', fail_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(2)

        except Exception as e:
            print(f"❌ Error during encoding capture: {e}")
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, f"Error: {e}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            ret_enc, buffer = cv2.imencode('.jpg', err_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(2)

        finally:
            with capture_lock:
                if video_capture:
                    video_capture.release()
                    video_capture = None
            print(f"ℹ️  Encoding capture stream closed.")

    def recognize_and_annotate_frame(self, frame):
        """Recognize faces using custom OpenCV implementation."""
        if not self.is_trained:
            cv2.putText(frame, "No faces trained", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.RESIZE_FACTOR, fy=self.RESIZE_FACTOR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        newly_recognized_ids = set()

        for (x, y, w, h) in faces:
            # Scale back up coordinates
            scale = 1 / self.RESIZE_FACTOR
            x_full = int(x * scale)
            y_full = int(y * scale)
            w_full = int(w * scale)
            h_full = int(h * scale)

            # Extract and preprocess face region from original frame
            face_roi = frame[y_full:y_full+h_full, x_full:x_full+w_full]
            
            if face_roi.size == 0:
                continue
                
            preprocessed_face = self.preprocess_face(face_roi)

            # Recognize face
            try:
                label, confidence = self.face_recognizer.predict(preprocessed_face)
                
                name_text = "Unknown"
                student_id = None
                color = (0, 0, 255)  # Red for Unknown

                if confidence < self.CONFIDENCE_THRESHOLD and label in self.label_to_id:
                    student_id = self.label_to_id[label]
                    name = self.id_to_name.get(student_id, "Unknown")
                    color = (0, 255, 0)  # Green for Known

                    # Voting Logic
                    with attendance_log_lock:
                        if student_id not in marked_today_set:
                            current_count = self.recognition_buffer.get(student_id, 0) + 1
                            self.recognition_buffer[student_id] = current_count
                            name_text = f"{name} ({current_count}/{self.RECOGNITION_THRESHOLD})"

                            if current_count >= self.RECOGNITION_THRESHOLD:
                                newly_recognized_ids.add(student_id)
                                self.recognition_buffer.pop(student_id, None)

                        elif student_id in marked_today_set:
                            name_text = f"{name} (Marked)"
                        else:
                            name_text = name
                else:
                    # Show confidence for debugging
                    name_text = f"Unknown ({confidence:.1f})"

            except Exception as e:
                name_text = "Recognition Error"
                color = (0, 0, 255)

            # Draw results
            cv2.rectangle(frame, (x_full, y_full), (x_full+w_full, y_full+h_full), color, 2)
            cv2.rectangle(frame, (x_full, y_full+h_full-25), (x_full+w_full, y_full+h_full), color, cv2.FILLED)
            cv2.putText(frame, name_text, (x_full+6, y_full+h_full-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Mark attendance in DB if threshold met
        if newly_recognized_ids:
            self.mark_attendance_in_db(newly_recognized_ids)

        return frame

    def mark_attendance_in_db(self, student_id_set):
        """Helper function to add attendance records to the DB"""
        print(f"ℹ️  Marking attendance for: {student_id_set}")
        today = datetime.date.today().isoformat()
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        try:
            with db.get_cursor() as cursor:
                for student_id in student_id_set:
                    existing = cursor.execute(
                        "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
                        (student_id, today)
                    ).fetchone()
                    if not existing:
                        cursor.execute(
                            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
                            (student_id, today, current_time, 'Present')
                        )
                        with attendance_log_lock:
                            marked_today_set.add(student_id)
                            print(f"✅ Successfully marked {student_id} in DB and set.")
                    else:
                        with attendance_log_lock:
                            if student_id not in marked_today_set:
                                marked_today_set.add(student_id)
                                print(f"ℹ️  {student_id} was already in DB, added to set for consistency.")

        except Exception as e:
            print(f"❌ Error marking attendance in DB: {e}")

face_system = AdvancedFaceRecognition()

# --- Webcam Streaming Generators ---

def gen_frames_capture(student_id):
    """Generator for the CAPTURE video feed (using custom OpenCV)."""
    yield from face_system.capture_and_save_face_encoding(student_id)
    print("ℹ️  Capture stream finished. Reloading trained data...")
    face_system.load_trained_data()

def gen_frames_recognize():
    """Generator for the RECOGNITION video feed (using custom OpenCV)."""
    global video_capture
    with capture_lock:
        if video_capture is None:
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                print("❌ Cannot open webcam for recognition")
                video_capture = None
                err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(err_frame, "Error: Cannot open webcam", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                ret_enc, buffer = cv2.imencode('.jpg', err_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return
        print("ℹ️  Recognition stream started.")

    try:
        while True:
            with capture_lock:
                if video_capture is None:
                    print("ℹ️  Video capture released externally. Stopping stream.")
                    break
                ret, frame = video_capture.read()

            if not ret:
                print("⚠️  Can't receive frame. Exiting recognition stream.")
                break

            # Process the frame using our custom recognizer
            annotated_frame = face_system.recognize_and_annotate_frame(frame)

            # Encode and yield
            ret_enc, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret_enc:
                print("⚠️ Error encoding frame")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"❌ Error in recognition stream: {e}")
    finally:
        with capture_lock:
            if video_capture:
                video_capture.release()
                video_capture = None
        print("ℹ️  Recognition stream closed.")

# --- Standard Flask Routes (Login, Logout) (Unchanged) ---
@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if not username:
            flash('Please enter username/ID', 'error')
            return render_login()

        if role == "admin":
            if not password:
                flash('Please enter password', 'error')
                return render_login()

            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            with db.get_cursor() as cursor:
                admin = cursor.execute(
                    "SELECT username FROM admins WHERE username = ? AND password = ?",
                    (username, hashed_password)
                ).fetchone()

            if admin:
                session['user_id'] = username
                session['role'] = 'admin'
                flash('Admin login successful!', 'success')
                return redirect('/admin/dashboard')
            else:
                flash('Invalid admin credentials', 'error')

        elif role == "teacher":
            if not password:
                flash('Please enter password', 'error')
                return render_login()

            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            with db.get_cursor() as cursor:
                teacher = cursor.execute(
                    "SELECT username FROM teachers WHERE username = ? AND password = ?",
                    (username, hashed_password)
                ).fetchone()

            if teacher:
                session['user_id'] = username
                session['role'] = 'teacher'
                flash('Login successful!', 'success')
                return redirect('/teacher/dashboard')
            else:
                flash('Invalid teacher credentials', 'error')

        else: # student
            with db.get_cursor() as cursor:
                student = cursor.execute(
                    "SELECT id, name FROM students WHERE id = ?", (username,)
                ).fetchone()

            if student:
                session['user_id'] = username
                session['role'] = 'student'
                session['student_name'] = student[1]
                flash('Login successful!', 'success')
                return redirect('/student/dashboard')
            else:
                flash('Invalid student ID', 'error')

    return render_login()

def render_login():
    login_content = '''
    <div class="row justify-content-center align-items-center min-vh-100">
        <div class="col-md-5 col-lg-4">
            <div class="glass-card p-5">
                <div class="text-center mb-4">
                    <i class="fas fa-graduation-cap fa-3x text-primary mb-3"></i>
                    <h2 class="fw-bold text-dark mb-2">Welcome Back</h2>
                    <p class="text-muted">Sign in to your account</p>
                </div>

                <form method="POST">
                    <div class="mb-3">
                        <label class="form-label fw-semibold">I am a:</label>
                        <select class="form-select border-0 py-3" name="role" required style="background: rgba(255,255,255,0.8);">
                            <option value="admin">Admin</option>
                            <option value="teacher">Teacher</option>
                            <option value="student">Student</option>
                        </select>
                    </div>

                    <div class="mb-3">
                        <label class="form-label fw-semibold">Username/ID:</label>
                        <input type="text" class="form-control border-0 py-3" name="username" required style="background: rgba(255,255,255,0.8);" placeholder="Enter your ID">
                    </div>

                    <div class="mb-4" id="password-field">
                        <label class="form-label fw-semibold">Password:</label>
                        <input type="password" class="form-control border-0 py-3" name="password" style="background: rgba(255,255,255,0.8);" placeholder="Enter password">
                    </div>

                    <button type="submit" class="btn btn-primary w-100 py-3 fw-semibold">
                        <i class="fas fa-sign-in-alt me-2"></i>Sign In
                    </button>
                </form>

                <div class="mt-4">
                    <div class="alert alert-info border-0">
                        <h6 class="fw-bold">Demo Credentials:</h6>
                        <p class="mb-1"><strong>Admin:</strong> admin / admin123</p>
                        <p class="mb-1"><strong>Teacher:</strong> teacher1 / pass123</p>
                        <p class="mb-0"><strong>Student:</strong> S1001, S1002</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.querySelector('select[name="role"]').addEventListener('change', function() {
            const role = this.value;
            const passwordField = document.getElementById('password-field');
            if (role === 'student') {
                passwordField.style.display = 'none';
            } else {
                passwordField.style.display = 'block';
            }
        });
        document.querySelector('select[name="role"]').dispatchEvent(new Event('change'));
    </script>
    '''
    return render_template_string(BASE_HTML, title="Login", content=login_content)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect('/login')

# --- Admin Routes (Unchanged) ---
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access', 'error')
        return redirect('/login')

    with db.get_cursor() as cursor:
        teacher_count = cursor.execute("SELECT COUNT(*) FROM teachers").fetchone()[0]
        student_count = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    dashboard_content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-shield-alt me-2"></i>Admin Dashboard</h1>
                <span class="badge bg-light text-dark fs-6">Welcome, {session['user_id']}</span>
            </div>
        </div>
    </div>
    <div class="row mb-4">
        <div class="col-md-6"><div class="stat-card"><div class="d-flex justify-content-between align-items-center"><div><h3 class="fw-bold">{teacher_count}</h3><p class="mb-0">Total Teachers</p></div><i class="fas fa-chalkboard-teacher fa-2x opacity-75"></i></div></div></div>
        <div class="col-md-6"><div class="stat-card"><div class="d-flex justify-content-between align-items-center"><div><h3 class="fw-bold">{student_count}</h3><p class="mb-0">Total Students</p></div><i class="fas fa-users fa-2x opacity-75"></i></div></div></div>
    </div>
    <div class="row"><div class="col-12"><div class="glass-card p-4"><h5 class="fw-bold mb-4"><i class="fas fa-cogs me-2"></i>Management Actions</h5><div class="row g-3">
        <div class="col-md-6"><a href="/admin/teachers" class="btn btn-outline-primary w-100 py-3 d-flex flex-column align-items-center"><i class="fas fa-chalkboard-teacher fa-2x mb-2"></i><span>Manage Teachers</span></a></div>
        <div class="col-md-6"><a href="/admin/students" class="btn btn-outline-success w-100 py-3 d-flex flex-column align-items-center"><i class="fas fa-users fa-2x mb-2"></i><span>Manage Students</span></a></div>
    </div></div></div></div>
    '''
    return render_template_string(BASE_HTML, title="Admin Dashboard", content=dashboard_content)

@app.route('/admin/teachers', methods=['GET', 'POST'])
def manage_teachers():
    if 'user_id' not in session or session.get('role') != 'admin': 
        return redirect('/login')
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password: 
            flash('Username and password are required', 'error')
        else:
            try:
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                with db.get_cursor() as cursor: 
                    cursor.execute("INSERT INTO teachers (username, password) VALUES (?, ?)", (username, hashed_password))
                flash(f'Teacher {username} added successfully!', 'success')
            except sqlite3.IntegrityError: 
                flash(f'Teacher {username} already exists', 'error')
        return redirect(url_for('manage_teachers'))
    
    with db.get_cursor() as cursor: 
        teachers = cursor.execute("SELECT id, username FROM teachers ORDER BY username").fetchall()
    
    content = '<div class="d-flex justify-content-between align-items-center mb-4"><h1 class="text-white fw-bold"><i class="fas fa-chalkboard-teacher me-2"></i>Manage Teachers</h1><a href="/admin/dashboard" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back</a></div><div class="row"><div class="col-md-8"><div class="glass-card p-4"><h5 class="fw-bold mb-4">All Teachers</h5><div class="table-responsive"><table class="table table-borderless table-hover"><thead><tr class="border-bottom"><th>ID</th><th>Username</th><th>Actions</th></tr></thead><tbody>'
    
    for teacher in teachers:
        delete_url = url_for('delete_teacher', teacher_id=teacher[0])
        content += f'''<tr>
            <td>{teacher[0]}</td>
            <td>{teacher[1]}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" data-bs-toggle="modal" data-bs-target="#editTeacherModal{teacher[0]}">
                    <i class="fas fa-edit me-1"></i>Edit
                </button>
                <form action="{delete_url}" method="POST" class="d-inline" onsubmit="return confirm('Delete this teacher?');">
                    <button type="submit" class="btn btn-sm btn-outline-danger">
                        <i class="fas fa-trash me-1"></i>Delete
                    </button>
                </form>
            </td>
        </tr>'''
    
    content += '</tbody></table></div></div></div><div class="col-md-4"><div class="glass-card p-4"><h5 class="fw-bold mb-4">Add New Teacher</h5><form method="POST"><div class="mb-3"><label class="form-label">Username</label><input type="text" name="username" class="form-control" required></div><div class="mb-3"><label class="form-label">Password</label><input type="password" name="password" class="form-control" required></div><button type="submit" class="btn btn-primary w-100">Add Teacher</button></form></div></div></div>'
    
    for teacher in teachers:
        edit_url = url_for('edit_teacher', teacher_id=teacher[0])
        content += f'''
        <div class="modal fade" id="editTeacherModal{teacher[0]}" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content glass-card">
                    <div class="modal-header">
                        <h5 class="modal-title">Edit Teacher: {teacher[1]}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <form action="{edit_url}" method="POST">
                        <div class="modal-body">
                            <div class="mb-3">
                                <label class="form-label">Username</label>
                                <input type="text" name="username" class="form-control" value="{teacher[1]}" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">New Password (optional)</label>
                                <input type="password" name="password" class="form-control" placeholder="Leave blank to keep old password">
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="submit" class="btn btn-primary">Save changes</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>'''
    
    return render_template_string(BASE_HTML, title="Manage Teachers", content=content)

@app.route('/admin/teacher/edit/<int:teacher_id>', methods=['POST'])
def edit_teacher(teacher_id):
    if 'user_id' not in session or session.get('role') != 'admin': 
        return redirect('/login')
    
    username = request.form['username']
    password = request.form['password']
    try:
        if password:
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            with db.get_cursor() as cursor: 
                cursor.execute("UPDATE teachers SET username = ?, password = ? WHERE id = ?", (username, hashed_password, teacher_id))
            flash(f'Teacher {username} updated successfully (with new password)!', 'success')
        else:
            with db.get_cursor() as cursor: 
                cursor.execute("UPDATE teachers SET username = ? WHERE id = ?", (username, teacher_id))
            flash(f'Teacher {username} updated successfully!', 'success')
    except sqlite3.IntegrityError: 
        flash(f'Teacher username {username} already exists', 'error')
    return redirect(url_for('manage_teachers'))

@app.route('/admin/teacher/delete/<int:teacher_id>', methods=['POST'])
def delete_teacher(teacher_id):
    if 'user_id' not in session or session.get('role') != 'admin': 
        return redirect('/login')
    
    try:
        with db.get_cursor() as cursor: 
            cursor.execute("DELETE FROM teachers WHERE id = ?", (teacher_id,))
        flash('Teacher deleted successfully!', 'success')
    except Exception as e: 
        flash(f'Error deleting teacher: {e}', 'error')
    return redirect(url_for('manage_teachers'))

@app.route('/admin/students', methods=['GET', 'POST'])
def manage_students():
    if 'user_id' not in session or session.get('role') != 'admin': 
        return redirect('/login')
    
    if request.method == 'POST':
        student_id = request.form['id']
        name = request.form['name']
        department = request.form['department']
        year = request.form['year']
        contact = request.form['contact']
        if not all([student_id, name, department, year, contact]): 
            flash('All fields are required', 'error')
        else:
            try:
                with db.get_cursor() as cursor: 
                    cursor.execute("INSERT INTO students (id, name, department, year, contact) VALUES (?, ?, ?, ?, ?)",
                                  (student_id, name, department, year, contact))
                flash(f'Student {name} added successfully!', 'success')
            except sqlite3.IntegrityError: 
                flash(f'Student ID {student_id} already exists', 'error')
        return redirect(url_for('manage_students'))
    
    with db.get_cursor() as cursor: 
        students = cursor.execute("SELECT id, name, department, year, contact FROM students ORDER BY id").fetchall()
    
    content = f'''
    <div class="d-flex justify-content-between align-items-center mb-4">
        <h1 class="text-white fw-bold"><i class="fas fa-users me-2"></i>Manage Students</h1>
        <a href="/admin/dashboard" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back</a>
    </div>
    <div class="row">
        <div class="col-md-12">
            <div class="glass-card p-4 mb-4">
                <h5 class="fw-bold mb-4">Add New Student</h5>
                <form method="POST" class="row g-3">
                    <div class="col-md-2"><label class="form-label">Student ID</label><input type="text" name="id" class="form-control" required></div>
                    <div class="col-md-3"><label class="form-label">Name</label><input type="text" name="name" class="form-control" required></div>
                    <div class="col-md-2"><label class="form-label">Department</label><input type="text" name="department" class="form-control" required></div>
                    <div class="col-md-2"><label class="form-label">Year</label><input type="text" name="year" class="form-control" required></div>
                    <div class="col-md-2"><label class="form-label">Contact</label><input type="text" name="contact" class="form-control" required></div>
                    <div class="col-md-1 d-flex align-items-end"><button type="submit" class="btn btn-primary w-100">Add</button></div>
                </form>
            </div>
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4">All Students</h5>
                <div class="table-responsive">
                    <table class="table table-borderless table-hover">
                        <thead><tr class="border-bottom"><th>ID</th><th>Name</th><th>Department</th><th>Year</th><th>Contact</th><th>Actions</th></tr></thead>
                        <tbody>'''
    
    for s in students:
        delete_url = url_for('delete_student', student_id=s[0])
        content += f'''
        <tr>
            <td>{s[0]}</td>
            <td>{s[1]}</td>
            <td>{s[2]}</td>
            <td>{s[3]}</td>
            <td>{s[4]}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" data-bs-toggle="modal" data-bs-target="#editStudentModal{s[0]}">
                    <i class="fas fa-edit me-1"></i>Edit
                </button>
                <form action="{delete_url}" method="POST" class="d-inline" onsubmit="return confirm('Delete this student AND all their attendance records?');">
                    <button type="submit" class="btn btn-sm btn-outline-danger">
                        <i class="fas fa-trash me-1"></i>Delete
                    </button>
                </form>
            </td>
        </tr>'''
    
    content += '</tbody></table></div></div></div></div>'
    
    for s in students:
        edit_url = url_for('edit_student', student_id=s[0])
        content += f'''
        <div class="modal fade" id="editStudentModal{s[0]}" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content glass-card">
                    <div class="modal-header">
                        <h5 class="modal-title">Edit Student: {s[1]} ({s[0]})</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <form action="{edit_url}" method="POST">
                        <div class="modal-body">
                            <div class="row g-3">
                                <div class="col-md-6"><label class="form-label">Name</label><input type="text" name="name" class="form-control" value="{s[1]}" required></div>
                                <div class="col-md-6"><label class="form-label">Department</label><input type="text" name="department" class="form-control" value="{s[2]}" required></div>
                                <div class="col-md-6"><label class="form-label">Year</label><input type="text" name="year" class="form-control" value="{s[3]}" required></div>
                                <div class="col-md-6"><label class="form-label">Contact</label><input type="text" name="contact" class="form-control" value="{s[4]}" required></div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                            <button type="submit" class="btn btn-primary">Save changes</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>'''
    
    return render_template_string(BASE_HTML, title="Manage Students", content=content)

@app.route('/admin/student/edit/<student_id>', methods=['POST'])
def edit_student(student_id):
    if 'user_id' not in session or session.get('role') != 'admin': 
        return redirect('/login')
    
    name = request.form['name']
    department = request.form['department']
    year = request.form['year']
    contact = request.form['contact']
    try:
        with db.get_cursor() as cursor: 
            cursor.execute("UPDATE students SET name = ?, department = ?, year = ?, contact = ? WHERE id = ?", 
                          (name, department, year, contact, student_id))
        flash(f'Student {name} updated successfully!', 'success')
    except Exception as e: 
        flash(f'Error updating student: {e}', 'error')
    return redirect(url_for('manage_students'))

@app.route('/admin/student/delete/<student_id>', methods=['POST'])
def delete_student(student_id):
    if 'user_id' not in session or session.get('role') != 'admin': 
        return redirect('/login')
    
    try:
        with db.get_cursor() as cursor: 
            cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        flash('Student and all related attendance records deleted successfully!', 'success')
    except Exception as e: 
        flash(f'Error deleting student: {e}', 'error')
    return redirect(url_for('manage_students'))


# --- Teacher Routes ---

@app.route('/teacher/dashboard')
def teacher_dashboard():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return redirect('/login')
    
    with db.get_cursor() as cursor:
        total_students = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        today = datetime.date.today().isoformat()
        today_attendance = cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today,)).fetchone()[0]
        trained_faces = cursor.execute("SELECT COUNT(*) FROM students WHERE face_encoding IS NOT NULL").fetchone()[0]
    
    dashboard_content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-tachometer-alt me-2"></i>Teacher Dashboard</h1>
                <span class="badge bg-light text-dark fs-6">Welcome, {session['user_id']}</span>
            </div>
        </div>
    </div>
    <div class="row mb-4">
        <div class="col-md-4">
            <div class="stat-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div><h3 class="fw-bold">{total_students}</h3><p class="mb-0">Total Students</p></div>
                    <i class="fas fa-users fa-2x opacity-75"></i>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stat-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div><h3 class="fw-bold">{today_attendance}</h3><p class="mb-0">Today's Attendance</p></div>
                    <i class="fas fa-clipboard-check fa-2x opacity-75"></i>
                </div>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stat-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div><h3 class="fw-bold">{trained_faces}</h3><p class="mb-0">Faces Trained</p></div>
                    <i class="fas fa-user-check fa-2x opacity-75"></i>
                </div>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-12">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4"><i class="fas fa-rocket me-2"></i>Quick Actions</h5>
                <div class="row g-3">
                    <div class="col-md-3">
                        <a href="/teacher/students" class="btn btn-outline-primary w-100 py-3 d-flex flex-column align-items-center">
                            <i class="fas fa-users fa-2x mb-2"></i><span>Students</span>
                        </a>
                    </div>
                    <div class="col-md-3">
                        <a href="/teacher/facial-attendance" class="btn btn-outline-success w-100 py-3 d-flex flex-column align-items-center">
                            <i class="fas fa-camera fa-2x mb-2"></i><span>Facial Attendance</span>
                        </a>
                    </div>
                    <div class="col-md-3">
                        <a href="/teacher/reports" class="btn btn-outline-info w-100 py-3 d-flex flex-column align-items-center">
                            <i class="fas fa-chart-bar fa-2x mb-2"></i><span>Reports</span>
                        </a>
                    </div>
                    <div class="col-md-3">
                        <a href="/teacher/train-faces" class="btn btn-outline-warning w-100 py-3 d-flex flex-column align-items-center">
                            <i class="fas fa-brain fa-2x mb-2"></i><span>Train Face</span>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>'''
    return render_template_string(BASE_HTML, title="Teacher Dashboard", content=dashboard_content)

@app.route('/teacher/students')
def view_students():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return redirect('/login')
    
    with db.get_cursor() as cursor: 
        students = cursor.execute("SELECT id, name, department, year, contact, face_encoding FROM students ORDER BY id").fetchall()
    
    content = '''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-users me-2"></i>Students</h1>
                <a href="/teacher/dashboard" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back</a>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-12">
            <div class="glass-card p-4">
                <div class="table-responsive">
                    <table class="table table-borderless table-hover">
                        <thead>
                            <tr class="border-bottom">
                                <th class="fw-semibold">ID</th>
                                <th class="fw-semibold">Name</th>
                                <th class="fw-semibold">Dept</th>
                                <th class="fw-semibold">Year</th>
                                <th class="fw-semibold">Contact</th>
                                <th class="fw-semibold">Face Trained</th>
                            </tr>
                        </thead>
                        <tbody>'''
    
    for s in students:
        face_status = '<span class="badge bg-success"><i class="fas fa-check me-1"></i>Yes</span>' if s[5] else '<span class="badge bg-secondary"><i class="fas fa-times me-1"></i>No</span>'
        content += f'''
        <tr>
            <td class="fw-bold text-primary">{s[0]}</td>
            <td>{s[1]}</td>
            <td><span class="badge bg-info">{s[2]}</span></td>
            <td><span class="badge bg-dark">{s[3]}</span></td>
            <td>{s[4]}</td>
            <td>{face_status}</td>
        </tr>'''
    
    content += '''
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>'''
    return render_template_string(BASE_HTML, title="Students", content=content)

# --- Training Routes (Custom OpenCV) ---

@app.route('/teacher/train-faces')
def train_faces():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return redirect('/login')
    
    with db.get_cursor() as cursor: 
        students = cursor.execute("SELECT id, name, face_encoding FROM students ORDER BY id").fetchall()
    trained_count = sum(1 for s in students if s[2])

    content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-brain me-2"></i>Train Face Encoding</h1>
                <a href="/teacher/dashboard" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back</a>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-md-6">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4">Student List ({trained_count}/{len(students)} Trained)</h5>
                <div class="list-group">'''
    
    for s in students:
        status_icon = '<i class="fas fa-check-circle text-success me-2"></i>' if s[2] else '<i class="fas fa-times-circle text-secondary me-2"></i>'
        btn_text = 'Re-train Face' if s[2] else 'Train Face'
        content += f'''
        <div class="list-group-item border-0 mb-2 rounded-3" style="background: rgba(255,255,255,0.8);">
            <div class="d-flex justify-content-between align-items-center">
                <div>{status_icon}<span class="fw-semibold">{s[0]}</span> - {s[1]}</div>
                <a href="/teacher/capture/{s[0]}" class="btn btn-sm btn-primary">
                    <i class="fas fa-camera me-1"></i>{btn_text}
                </a>
            </div>
        </div>'''
    
    content += '''
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4">Training Instructions (OpenCV Custom)</h5>
                <div class="alert alert-info border-0">
                    <h6><i class="fas fa-info-circle me-2"></i>How to train:</h6>
                    <ul class="mb-0">
                        <li>Click "Train Face" for a student.</li>
                        <li>Look directly at the camera with a neutral expression.</li>
                        <li>The system will capture <strong>10 face samples</strong> automatically.</li>
                        <li>Hold still and ensure good lighting for best results.</li>
                        <li>At least 5 samples are required for successful training.</li>
                        <li>Multiple samples improve recognition accuracy.</li>
                    </ul>
                </div>
                <div class="alert alert-warning border-0">
                    <h6><i class="fas fa-exclamation-triangle me-2"></i>Notes:</h6>
                    <ul class="mb-0">
                        <li>This uses OpenCV's LBPH face recognizer (custom implementation).</li>
                        <li>Works well with good lighting and frontal faces.</li>
                        <li>More samples = better recognition accuracy.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>'''
    return render_template_string(BASE_HTML, title="Train Faces", content=content)

@app.route('/teacher/capture/<student_id>')
def capture_page(student_id):
    """Page to host the video capture feed for custom OpenCV encoding."""
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return redirect('/login')
    
    with db.get_cursor() as cursor: 
        student = cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,)).fetchone()
    if not student: 
        flash('Student not found', 'error')
        return redirect('/teacher/train-faces')
    
    student_name = student[0]

    content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-camera me-2"></i>Capture Face Samples: {student_name}</h1>
                <a href="/teacher/train-faces" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back to List</a>
            </div>
        </div>
    </div>
    <div class="row justify-content-center">
        <div class="col-lg-8">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-3 text-center">Capturing 10 face samples for training...</h5>
                <div class="camera-feed mb-3">
                    <img id="video-stream" src="{url_for('video_feed_capture', student_id=student_id)}" alt="Loading webcam..." />
                </div>
                <div class="alert alert-info text-center">
                    Please look directly at the camera. The system will automatically capture 10 samples.
                    <br>This page will redirect automatically after completion.
                </div>
            </div>
        </div>
    </div>
    <script>
        const video = document.getElementById('video-stream');
        let notificationShown = false;

        video.addEventListener('error', function() {{
            if (!notificationShown) {{
                showNotification('Capture process finished. Reloading data...', 'info');
                notificationShown = true;
                setTimeout(() => {{
                    window.location.href = '/teacher/train-faces';
                }}, 2500);
            }}
        }});
        
        video.addEventListener('load', function() {{
            console.log("Video stream started.");
        }});

        function showNotification(message, type) {{
            const notification = document.createElement('div');
            notification.className = `alert alert-${{type === 'error' ? 'danger' : type}}`;
            notification.style.position = 'fixed'; 
            notification.style.top = '20px';
            notification.style.right = '20px'; 
            notification.style.zIndex = '9999';
            notification.innerHTML = message;
            document.body.appendChild(notification);
            setTimeout(() => {{ if(notification.parentNode) notification.remove(); }}, 3000);
        }}
    </script>
    '''
    return render_template_string(BASE_HTML, title="Capture Face Samples", content=content)

@app.route('/video_feed_capture/<student_id>')
def video_feed_capture(student_id):
    """Video streaming route for capturing face samples."""
    return Response(gen_frames_capture(student_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Recognition Routes ---

@app.route('/teacher/facial-attendance')
def facial_attendance():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return redirect('/login')

    # Reload data when visiting page
    face_system.load_trained_data()

    with db.get_cursor() as cursor:
        trained_students = len(face_system.label_to_id)
        total_students = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    face_system.recognition_buffer.clear()

    # Reset today's marked set
    with attendance_log_lock:
        marked_today_set.clear()
        today = datetime.date.today().isoformat()
        with db.get_cursor() as cursor:
            attended = cursor.execute("SELECT DISTINCT student_id FROM attendance WHERE date = ?", (today,)).fetchall()
            for (student_id,) in attended: 
                marked_today_set.add(student_id)

    content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-camera me-2"></i>MARK ATTENDANCE</h1>
                <a href="/teacher/dashboard" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back</a>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-lg-8">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4">Live Recognition Feed</h5>
                <div class="camera-feed mb-4" id="camera-feed">
                    <div class="text-center" id="camera-placeholder">
                        <i class="fas fa-camera fa-3x mb-3 text-muted"></i>
                        <h5>Recognition System Offline</h5>
                        <p class="mb-0">{trained_students}/{total_students} students trained</p>
                        {"<div class='alert alert-warning text-dark mt-3'>Training at least one face is required.</div>" if trained_students == 0 else ""}
                    </div>
                </div>
                <div class="d-flex gap-2 justify-content-center flex-wrap mb-4">
                    <button onclick="startFacialRecognition()" class="btn btn-success px-4" id="start-btn" {"disabled" if not face_system.is_trained else ""}>
                        <i class="fas fa-play me-2"></i>Start Recognition
                    </button>
                    <button onclick="stopFacialRecognition()" class="btn btn-danger px-4" id="stop-btn" style="display: none;">
                        <i class="fas fa-stop me-2"></i>Stop Recognition
                    </button>
                    <button onclick="markManualAttendance()" class="btn btn-primary px-4">
                        <i class="fas fa-user-plus me-2"></i>Manual Entry
                    </button>
                </div>
            </div>
        </div>
        <div class="col-lg-4">
            <div class="glass-card p-4 mb-4">
                <h5 class="fw-bold mb-4">Attendance Log (Live)</h5>
                <div id="attendance-log" style="max-height: 400px; overflow-y: auto;">
                    <div class="text-center text-muted py-4" id="log-placeholder">
                        <i class="fas fa-clock fa-2x mb-2"></i>
                        <p>No attendance marked yet</p>
                    </div>
                </div>
            </div>
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-3">Today's Summary</h5>
                <div class="text-center">
                    <h2 id="today-count" class="text-primary fw-bold">{len(marked_today_set)}</h2>
                    <p class="text-muted mb-0">Students marked today</p>
                </div>
                <div class="mt-3">
                    <button onclick="clearTodayAttendance()" class="btn btn-outline-danger btn-sm w-100">
                        <i class="fas fa-trash me-1"></i>Clear Today's Attendance
                    </button>
                </div>
            </div>
        </div>
    </div>
<script>
let logPoller;
let knownLogEntries = new Set();

function startFacialRecognition() {{
    console.log("Starting facial recognition...");
    
    // Hide start button, show stop button
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'inline-block';
    
    // Hide placeholder
    document.getElementById('camera-placeholder').style.display = 'none';
    
    // Create video stream element
    const cameraFeed = document.getElementById('camera-feed');
    cameraFeed.innerHTML = '<img id="video-stream" src="/video_feed_recognize" style="width:100%; height:auto; border-radius:16px;">';
    
    // Start polling for attendance updates
    loadInitialLog();
    logPoller = setInterval(updateAttendanceLog, 2000);
    
    showNotification('Facial recognition started! Looking for faces...', 'success');
}}

function stopFacialRecognition() {{
    console.log("Stopping facial recognition...");
    
    // Show start button, hide stop button
    document.getElementById('start-btn').style.display = 'inline-block';
    document.getElementById('stop-btn').style.display = 'none';
    
    // Show placeholder
    document.getElementById('camera-placeholder').style.display = 'block';
    
    // Clear video stream
    const cameraFeed = document.getElementById('camera-feed');
    cameraFeed.innerHTML = '<div class="text-center" id="camera-placeholder"><i class="fas fa-camera fa-3x mb-3 text-muted"></i><h5>Recognition System Offline</h5></div>';
    
    // Stop polling
    if (logPoller) {{
        clearInterval(logPoller);
    }}
    
    showNotification('Facial recognition stopped.', 'warning');
}}

function loadInitialLog() {{
    knownLogEntries.clear();
    updateAttendanceLog();
}}

function updateAttendanceLog() {{
    fetch('/api/today_attendance_log')
        .then(response => response.json())
        .then(data => {{
            if (data.success) {{
                document.getElementById('today-count').textContent = data.attendance.length;
                
                const logDiv = document.getElementById('attendance-log');
                if (data.attendance.length === 0) {{
                    logDiv.innerHTML = '<div class="text-center text-muted py-4"><i class="fas fa-clock fa-2x mb-2"></i><p>No attendance marked yet</p></div>';
                }} else {{
                    let html = '';
                    data.attendance.forEach(entry => {{
                        if (!knownLogEntries.has(entry.id)) {{
                            html = '<div class="alert alert-success border-0 mb-2"><div class="d-flex justify-content-between align-items-center"><div><strong>' + 
                                   entry.name + '</strong><br><small class="text-muted">' + entry.id + ' • ' + entry.time + 
                                   '</small></div><i class="fas fa-check-circle text-success"></i></div></div>' + html;
                            knownLogEntries.add(entry.id);
                        }}
                    }});
                    if (html) {{
                        const placeholder = document.getElementById('log-placeholder');
                        if (placeholder) placeholder.remove();
                        logDiv.innerHTML = html + logDiv.innerHTML;
                    }}
                }}
            }}
        }})
        .catch(error => console.error('Error fetching attendance:', error));
}}

function markManualAttendance() {{
    const studentId = prompt('Enter Student ID:');
    if (studentId) {{
        fetch('/api/mark-attendance/' + studentId.trim())
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    showNotification('Attendance marked for ' + data.student_name, 'success');
                    updateAttendanceLog();
                }} else {{
                    showNotification(data.message, 'error');
                }}
            }});
    }}
}}

function clearTodayAttendance() {{
    if (confirm('Clear all attendance for today?')) {{
        fetch('/api/clear-today-attendance', {{ method: 'POST' }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    showNotification('Attendance cleared!', 'success');
                    knownLogEntries.clear();
                    updateAttendanceLog();
                }}
            }});
    }}
}}

function showNotification(message, type) {{
    // Use Bootstrap's built-in alert system through flashed messages
    alert(message); // Simple alert for now
}}

// Load initial attendance data
updateAttendanceLog();
</script>
    '''
    return render_template_string(BASE_HTML, title="Facial Attendance", content=content)

@app.route('/video_feed_recognize')
def video_feed_recognize():
    """Video streaming route for recognition using custom OpenCV."""
    return Response(gen_frames_recognize(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Routes ---
@app.route('/api/today_attendance_log')
def api_today_attendance_log():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    try:
        today = datetime.date.today().isoformat()
        with db.get_cursor() as cursor:
            records = cursor.execute('''
                SELECT a.student_id, s.name, a.time
                FROM attendance a JOIN students s ON a.student_id = s.id
                WHERE a.date = ? ORDER BY a.time DESC, a.id DESC
            ''', (today,)).fetchall()
        attendance_list = [{'id': r[0], 'name': r[1], 'time': r[2]} for r in records]
        return jsonify({'success': True, 'attendance': attendance_list})
    except Exception as e: 
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/student-info/<student_id>')
def api_student_info(student_id):
    try:
        with db.get_cursor() as cursor: 
            student = cursor.execute("SELECT id, name FROM students WHERE id = ?", (student_id,)).fetchone()
        if student: 
            return jsonify({'exists': True, 'id': student[0], 'name': student[1]})
        else: 
            return jsonify({'exists': False})
    except Exception as e: 
        return jsonify({'exists': False, 'error': str(e)})

@app.route('/api/mark-attendance/<student_id>')
def api_mark_attendance(student_id):
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    try:
        with db.get_cursor() as cursor: 
            student = cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,)).fetchone()
        if not student: 
            return jsonify({'success': False, 'message': 'Student not found'})
        
        student_name = student[0]
        today = datetime.date.today().isoformat()
        
        with attendance_log_lock:
            with db.get_cursor() as cursor: 
                existing = cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, today)).fetchone()
            if not existing:
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                with db.get_cursor() as cursor: 
                    cursor.execute("INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)", 
                                  (student_id, today, current_time, 'Present'))
                marked_today_set.add(student_id)
                return jsonify({'success': True, 'student_name': student_name, 'student_id': student_id, 'time': current_time})
            else:
                if student_id not in marked_today_set: 
                    marked_today_set.add(student_id)
                return jsonify({'success': False,'message': f'Attendance already marked for {student_name} ({student_id}) today'})
    except Exception as e: 
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/today-attendance')
def api_today_attendance():
    try:
        today = datetime.date.today().isoformat()
        with db.get_cursor() as cursor: 
            count = cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today,)).fetchone()[0]
        return jsonify({'count': count})
    except Exception as e: 
        return jsonify({'count': 0, 'error': str(e)})

@app.route('/api/clear-today-attendance', methods=['POST'])
def api_clear_today_attendance():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    try:
        today = datetime.date.today().isoformat()
        with db.get_cursor() as cursor: 
            cursor.execute("DELETE FROM attendance WHERE date = ?", (today,))
        with attendance_log_lock: 
            marked_today_set.clear()
        face_system.recognition_buffer.clear()
        return jsonify({'success': True, 'message': 'Today\'s attendance cleared successfully'})
    except Exception as e: 
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

# --- Reports and Student Dashboard (Unchanged) ---
@app.route('/teacher/reports')
def view_reports():
    if 'user_id' not in session or session.get('role') != 'teacher': 
        return redirect('/login')
    
    with db.get_cursor() as cursor:
        attendance_data = cursor.execute('SELECT s.name, s.id, a.date, a.time, a.status FROM attendance a JOIN students s ON a.student_id = s.id ORDER BY a.date DESC, a.time DESC LIMIT 100').fetchall()
        stats = cursor.execute('SELECT (SELECT COUNT(DISTINCT id) FROM students), COUNT(*), COUNT(DISTINCT date) FROM attendance').fetchone()
        avg_daily_result = cursor.execute('SELECT AVG(daily_count) FROM (SELECT COUNT(DISTINCT student_id) as daily_count FROM attendance GROUP BY date)').fetchone()
        avg_daily = avg_daily_result[0] if avg_daily_result and avg_daily_result[0] is not None else 0

    content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-chart-bar me-2"></i>Reports & Analytics</h1>
                <a href="/teacher/dashboard" class="btn btn-light"><i class="fas fa-arrow-left me-2"></i>Back</a>
            </div>
        </div>
    </div>
    <div class="row mb-4">
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{stats[0]}</h3><p class="mb-0">Total Students</p></div></div></div>
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{stats[1]}</h3><p class="mb-0">Total Records</p></div></div></div>
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{stats[2]}</h3><p class="mb-0">Total Days</p></div></div></div>
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{avg_daily:.1f}</h3><p class="mb-0">Avg Daily</p></div></div></div>
    </div>
    <div class="row">
        <div class="col-12">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4">Recent Attendance Records</h5>'''
    
    if attendance_data:
        content += '<div class="table-responsive"><table class="table table-borderless table-hover"><thead><tr class="border-bottom"><th class="fw-semibold">Name</th><th class="fw-semibold">ID</th><th class="fw-semibold">Date</th><th class="fw-semibold">Time</th><th class="fw-semibold">Status</th></tr></thead><tbody>'
        for r in attendance_data: 
            status_badge = 'bg-success' if r[4] == 'Present' else 'bg-warning'
            status_text = 'Present' if r[4] == 'Present' else 'Absent'
            content += f'<tr><td>{r[0]}</td><td><span class="badge bg-secondary">{r[1]}</span></td><td>{r[2]}</td><td>{r[3]}</td><td><span class="badge {status_badge}">{status_text}</span></td></tr>'
        content += '</tbody></table></div>'
    else: 
        content += '<div class="text-center py-5"><i class="fas fa-inbox fa-3x text-muted mb-3"></i><h5 class="text-muted">No Attendance Records</h5><p class="text-muted">No attendance recorded yet.</p></div>'
    
    content += '</div></div></div>'
    return render_template_string(BASE_HTML, title="Reports", content=content)

@app.route('/student/dashboard')
def student_dashboard():
    if 'user_id' not in session or session.get('role') != 'student': 
        return redirect('/login')
    
    student_id = session['user_id']
    with db.get_cursor() as cursor:
        student_info = cursor.execute("SELECT name, department, year, contact FROM students WHERE id = ?", (student_id,)).fetchone()
        records = cursor.execute("SELECT date, status FROM attendance WHERE student_id = ? ORDER BY date DESC", (student_id,)).fetchall()
    
    if not student_info: 
        flash('Student info not found', 'error')
        return redirect('/logout')
    
    student_name, department, year, contact = student_info
    total_classes = len(records)
    present_count = len([r for r in records if r[1] == 'Present'])
    percentage = (present_count / total_classes * 100) if total_classes > 0 else 0
    recent_attendance = records[:10]
    
    content = f'''
    <div class="row">
        <div class="col-12">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1 class="text-white fw-bold"><i class="fas fa-tachometer-alt me-2"></i>Student Dashboard</h1>
                <span class="badge bg-light text-dark">Welcome, {student_name}</span>
            </div>
        </div>
    </div>
    <div class="row mb-4">
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{total_classes}</h3><p class="mb-0">Total Classes</p></div></div></div>
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{present_count}</h3><p class="mb-0">Present</p></div></div></div>
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{total_classes - present_count}</h3><p class="mb-0">Absent</p></div></div></div>
        <div class="col-md-3"><div class="stat-card"><div class="text-center"><h3 class="fw-bold">{percentage:.1f}%</h3><p class="mb-0">Percentage</p></div></div></div>
    </div>
    <div class="row">
        <div class="col-md-6">
            <div class="glass-card p-4 mb-4">
                <h5 class="fw-bold mb-4"><i class="fas fa-user me-2"></i>Student Information</h5>
                <div class="row">
                    <div class="col-6">
                        <p><strong>ID:</strong><br>{student_id}</p>
                        <p><strong>Name:</strong><br>{student_name}</p>
                    </div>
                    <div class="col-6">
                        <p><strong>Dept:</strong><br>{department}</p>
                        <p><strong>Year:</strong><br>{year}</p>
                    </div>
                </div>
            </div>
            <div class="glass-card p-4">'''
    
    if percentage < 75 and total_classes > 0:
        content += f'<div class="alert alert-danger border-0"><h5 class="fw-bold"><i class="fas fa-exclamation-triangle me-2"></i>Attendance Warning</h5><p class="mb-0">Your attendance is {percentage:.1f}%, below the required 75%.</p></div>'
    elif total_classes > 0:
        content += f'<div class="alert alert-success border-0"><h5 class="fw-bold"><i class="fas fa-check-circle me-2"></i>Good Standing</h5><p class="mb-0">Your attendance is {percentage:.1f}%. Keep it up!</p></div>'
    else:
        content += f'<div class="alert alert-info border-0"><h5 class="fw-bold"><i class="fas fa-info-circle me-2"></i>Welcome!</h5><p class="mb-0">No attendance records found yet.</p></div>'
    
    content += '''
            </div>
        </div>
        <div class="col-md-6">
            <div class="glass-card p-4">
                <h5 class="fw-bold mb-4"><i class="fas fa-history me-2"></i>Recent Attendance</h5>'''
    
    if recent_attendance:
        content += '<div class="table-responsive"><table class="table table-borderless table-sm"><thead><tr><th class="fw-semibold">Date</th><th class="fw-semibold">Status</th></tr></thead><tbody>'
        for r in recent_attendance: 
            status_badge = 'bg-success' if r[1] == 'Present' else 'bg-danger'
            status_text = 'Present' if r[1] == 'Present' else 'Absent'
            content += f'<tr><td>{r[0]}</td><td><span class="badge {status_badge}">{status_text}</span></td></tr>'
        content += '</tbody></table></div>'
    else: 
        content += '<div class="text-center py-4"><i class="fas fa-clock fa-2x text-muted mb-3"></i><p class="text-muted mb-0">No records available</p></div>'
    
    content += '</div></div></div>'
    return render_template_string(BASE_HTML, title="Student Dashboard", content=content)

# --- Main Execution ---

def find_available_port(start_port=5000, max_port=5010):
    import socket
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
                s.bind(('127.0.0.1', port))
                return port
        except OSError: 
            continue
    return start_port

if __name__ == '__main__':
    print("=" * 60)
    print("🎓 EduFace - OpenCV Custom Face Recognition Attendance System")
    print("=" * 60)
    print("📊 Database initialized (students_opencv_custom.db)")
    print("🤖 Using custom OpenCV implementation (LBPH)")
    print("⚙️  Features:")
    print("   - Haar Cascade for face detection")
    print("   - LBPH (Local Binary Patterns Histograms) for recognition")
    print("   - Multiple sample capture for better accuracy")
    print("   - Advanced preprocessing (histogram equalization, blur)")
    print(f"📈 Loaded {len(face_system.label_to_id)} known students.")
    print("\n⚠️  IMPORTANT:")
    print("   - No external face_recognition/dlib dependencies!")
    print("   - Pure OpenCV implementation")
    print("   - Captures 10 face samples per student for robust recognition")
    print("\n👨‍💻 Admin: admin / admin123")
    print("👨‍🏫 Teacher: teacher1 / pass123")
    print("👨‍🎓 Student: S1001, S1002, ... (no password)")
    print("=" * 60)

    port = find_available_port(5000, 5010)
    print(f"\n🌐 Starting server at: http://127.0.0.1:{port}")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)
