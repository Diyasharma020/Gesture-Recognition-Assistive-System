# Gesture Recognition Assistive Communication System

**Python | OpenCV | MediaPipe | Streamlit**

A real-time hand gesture recognition system that detects commonly used hand gestures and converts them into meaningful English phrases. The application is designed to assist individuals with speech impairments by enabling fast, intuitive, and contactless communication using only a standard webcam.

---

## Problem Statement

Individuals with speech impairments often rely on sign language or hand gestures to communicate, which may not always be understood by others. This project provides an AI-powered solution that automatically recognizes predefined hand gestures in real time and translates them into simple English phrases, improving accessibility and communication.

---

## System Architecture

The project follows a modular architecture for improved scalability and maintainability.

### 1. Camera Module

* Captures real-time video frames using OpenCV.
* Streams frames continuously to the gesture detection pipeline.

### 2. Gesture Detection Module

* Uses MediaPipe Hands to detect 21 hand landmarks.
* Classifies gestures using geometric rules based on finger positions, landmark distances, and joint angles.

### 3. User Interface Module (`app.py`)

* Built using Streamlit.
* Displays live webcam feed.
* Highlights detected gestures.
* Provides adjustable detection thresholds.
* Allows users to save snapshots.

### 4. Utility Module

* Optional text-to-speech support using `pyttsx3`.
* Snapshot capture and image saving utilities.

---

## Working Algorithm

1. Capture a frame from the webcam.
2. Detect hand landmarks using MediaPipe's hand-tracking model.
3. Determine finger extension by comparing fingertip and PIP joint positions.
4. Compute distances and angles between landmarks to identify gesture patterns.
5. Apply a voting buffer across consecutive frames to eliminate prediction flickering.
6. Map the recognized gesture to its corresponding English phrase.
7. Display the detected phrase in real time through the Streamlit interface.

---

## Key Features

* Real-time hand gesture recognition
* MediaPipe-based 21-point hand landmark detection
* Supports **8 predefined gestures**

  * Help
  * Yes
  * No
  * OK
  * Wait
  * Thank You
  * I'm Okay
  * Call Me
* Interactive Streamlit web interface
* Live webcam preview
* Adjustable detection thresholds
* Optional text-to-speech output
* Snapshot saving functionality
* Lightweight and runs entirely offline
* Compatible with any standard laptop webcam

---

## Project Structure

```
Gesture-Recognition/
│
├── app.py                 # Streamlit application
├── gesture.py             # Gesture recognition engine
├── requirements.txt       # Python dependencies
├── screenshots/           # Sample outputs
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Gesture-Recognition
```

Install the required packages:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install opencv-python mediapipe streamlit pyttsx3
```

---

## Running the Project

### Method 1: Standalone Gesture Detector

```bash
python gesture.py
```

This will:

* Open the webcam
* Detect hand gestures in real time
* Display the corresponding English phrase

---

### Method 2: Streamlit Web Interface

```bash
streamlit run app.py
```

The Streamlit application provides:

* Live webcam feed
* Real-time gesture recognition
* Detection confidence tuning
* Snapshot capture
* Optional voice output

---

## Future Enhancements

* Deep learning-based gesture classification
* Support for complete sign language alphabets
* Multi-hand gesture recognition
* Speech-to-text and text-to-speech integration
* Mobile and web deployment
* Multilingual phrase translation

---

## Applications

* Assistive communication for individuals with speech impairments
* Smart healthcare systems
* Human-computer interaction
* Educational demonstrations
* Contactless gesture-controlled interfaces

---

## Screenshots-
<img width="1920" height="1080" alt="Screenshot (159)" src="https://github.com/user-attachments/assets/06231564-3e7d-4b1f-afd1-14b531126167" />
<img width="1920" height="1080" alt="Screenshot (160)" src="https://github.com/user-attachments/assets/8c7a70cd-824c-4305-b0bc-b15b09e68d9b" />
<img width="1920" height="1080" alt="Screenshot (161)" src="https://github.com/user-attachments/assets/f417aad1-c925-48a9-a59c-846304ee86da" />
<img width="1920" height="1080" alt="Screenshot (163)" src="https://github.com/user-attachments/assets/bb8300a7-3b60-48e6-abb0-a38eed308792" />
<img width="1920" height="1080" alt="Screenshot (164)" src="https://github.com/user-attachments/assets/b55a061b-76e2-4026-b47a-fc56603f58df" />
<img width="1920" height="1080" alt="Screenshot (165)" src="https://github.com/user-attachments/assets/6fc7c019-459f-4f9b-a107-5e92eed60bca" />
<img width="1920" height="1080" alt="Screenshot (166)" src="https://github.com/user-attachments/assets/330cdf8a-10d7-4eb5-935e-95d71f91e7c0" />
<img width="1920" height="1080" alt="Screenshot (167)" src="https://github.com/user-attachments/assets/2bbaef8e-5ce3-4652-8899-53052d49e6df" />
<img width="1920" height="1080" alt="Screenshot (168)" src="https://github.com/user-attachments/assets/edf83f26-23c9-4257-ae1b-62515d1c8564" />



Technologies Used- 
Python, OpenCV, MediaPipe Hands, Streamlit, NumPy, pyttsx3 (optional)
