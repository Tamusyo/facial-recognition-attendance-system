🚀 Project Overview

This system eliminates manual attendance marking by using computer vision and machine learning techniques. It captures facial data, trains a recognition model, and automatically marks attendance when a registered student is detected.

The application supports multiple user roles including Admin, Teacher, and Student with separate dashboards and features.

🛠️ Technologies Used

Python

Flask (Web Framework)

OpenCV (Face Detection & Recognition)

LBPH Face Recognizer

SQLite (Database)

HTML, CSS, JavaScript

Haar Cascade Classifier

✨ Features

✅ User Authentication (Admin / Teacher / Student)
✅ Add, Update, Delete Students
✅ Face Data Collection via Webcam
✅ Real-Time Face Recognition
✅ Automatic Attendance Marking
✅ Attendance Dashboard
✅ Attendance Reports
✅ REST API Endpoints
✅ Live Video Streaming

🧠 How It Works

Admin/Teacher registers students.

The system captures multiple face samples using a webcam.

Faces are trained using LBPH algorithm.

During recognition, the webcam detects faces.

If a match is found, attendance is automatically recorded in the database.

Attendance records can be viewed from the dashboard.

📂 Project Structure
PBL/
│── app.py
│── database file
│── static/
│── templates/
│── README.md

▶️ How to Run
pip install flask opencv-python numpy
python app.py


Then open:

http://127.0.0.1:5000

🎯 Use Cases

Schools & Colleges

Training Institutes

Organizations

Smart Classroom Systems

📌 Future Improvements

Cloud Deployment

Email Notifications

Export Attendance (CSV/Excel)

Improved Face Accuracy

Mobile App Integration

👨‍💻 Author

Developed as a Project Based Learning (PBL) project using Python and Computer Vision.

If you want, I can also:

🔥 Write a more professional industry-level README

📄 Create a shorter GitHub description (2–3 lines)

🏆 Make it resume-ready

📊 Add architecture diagram explanation

Tell me what style you want 🚀

in only 350 characters 

A web-based Facial Recognition Attendance System built with Python, Flask, OpenCV, and SQLite. It captures and trains facial data using a webcam, recognizes students in real time, and automatically marks attendance. Includes role-based dashboards for admin, teacher, and students with attendance tracking and reports.

give me readme file 

Here’s a clean and professional README.md file for your project 👇

You can copy-paste this into your GitHub repository.

🎓 Facial Recognition Attendance System

A web-based Facial Recognition Attendance System built using Python, Flask, OpenCV, and SQLite.
The system automatically detects and recognizes student faces in real time and marks attendance without manual intervention.

🚀 Features

🔐 Role-based login (Admin / Teacher / Student)

👤 Student management (Add / Update / Delete)

📸 Face data collection using webcam

🧠 Face training using LBPH algorithm

🎥 Real-time face recognition

✅ Automatic attendance marking

📊 Attendance dashboard & reports

🌐 REST API endpoints

📡 Live video streaming

🛠️ Tech Stack

Python

Flask

OpenCV

SQLite

NumPy

HTML / CSS / JavaScript

Haar Cascade Classifier

LBPH Face Recognizer

🧠 How It Works

Admin/Teacher registers students.

Face samples are captured through webcam.

The system trains a face recognition model.

During recognition, faces are matched with stored data.

Attendance is automatically recorded in the database.

Users can view attendance from their dashboard.

📂 Project Structure
PBL/
│── app.py
│── students_opencv_custom.db
│── static/
│── templates/
│── README.md

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Install dependencies
pip install flask opencv-python numpy

3️⃣ Run the application
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000

📌 Requirements

Python 3.x

Webcam

Internet browser

Basic Python environment

🎯 Use Cases

Schools & Colleges

Training Institutes

Smart Classrooms

Office Attendance Systems

🚀 Future Improvements

Cloud deployment

CSV/Excel export

Email notifications

Improved face recognition accuracy

Mobile app integration

👨‍💻 Author

Developed as a Project Based Learning (PBL) project using Python and Computer Vision.
