Gesture Recognition Assistive Communication System using Python, OpenCV & MediaPipe

A real-time hand-gesture detection system that recognizes 8 commonly used gestures and converts them into simple English phrases. Built to support individuals with speech impairments by providing a quick and intuitive communication method.

Problem Definition :
The system aims to assist individuals with speech impairments by providing an automatic method to interpret hand gestures. It detects eight predefined gestures through a webcam and converts them into meaningful English phrases for easier communication.

System Design & Modularization-
The project follows a modular structure:

• Camera Module: Captures real-time frames from the webcam.

• Detection Module: Uses MediaPipe to extract 21 hand landmarks and applies geometric rules to classify gestures.

• UI Module (app.py): Handles the Streamlit interface, displays video feed, highlights detected gestures, and provides tuning controls.

• Utility Functions: Provide optional text-to-speech and snapshot saving.

Algorithm Summary-

1)Capture a video frame from the webcam.

2)Detect hand landmarks using MediaPipe’s hand-tracking model.

3)Determine finger extension by comparing tip and pip positions.

4)Calculate distances and angles between landmarks to identify specific gestures.

5)Use a voting buffer to avoid flickering and ensure stable predictions.

6)Map the detected gesture to a predefined English phrase and display it.

Features-
• Real-time gesture detection using OpenCV and MediaPipe

• Supports 8 meaningful gestures (Help, Yes, No, OK, Wait, Thank You, I’m Okay, Call Me)

• Streamlit web app for clean UI and live preview

• Optional text-to-speech output

• Works fully offline

• Accurate detection with tuning sliders

• Lightweight, runs on any laptop webcam

Project Structure:

• app.py (Streamlit real-time gesture interface)

• gesture.py (Standalone gesture detection script)

• requirements.txt

• screenshots/ (Output images and system demonstration) 

• README.md

Requirements
Install dependencies:
pip install opencv-python

pip install mediapipe

pip install streamlit

pip install pyttsx3 (optional for voice output)

Usage-

Run the standalone detector:
python gesture.py-

• Opens webcam

• Detects gestures

• Displays recognized phrase

Run the Streamlit interface:
python -m streamlit run app.py -

• Shows live video preview

• Highlights detected gestures

• Lets you adjust thresholds and tuning

• Allows snapshot saving

• Supports optional text-to-speech

Technologies Used- 
Python, OpenCV, MediaPipe Hands, Streamlit, NumPy, pyttsx3 (optional)
