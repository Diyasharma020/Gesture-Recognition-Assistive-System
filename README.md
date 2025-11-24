Gesture Recognition Assistive Communication System using Python, OpenCV & MediaPipe

A real-time hand-gesture detection system that recognizes 8 commonly used gestures and converts them into simple English phrases. Built to support individuals with speech impairments by providing a quick and intuitive communication method.

Features
• Real-time gesture detection using OpenCV and MediaPipe

• Supports 8 meaningful gestures (Help, Yes, No, OK, Wait, Thank You, I’m Okay, Call Me)

• Streamlit web app for clean UI and live preview

• Optional text-to-speech output

• Works fully offline

• Accurate detection with tuning sliders

• Lightweight, runs on any laptop webcam

Project Structure:

gesture detection-project

• app.py (Streamlit real-time gesture interface)

• gesture.py (Standalone gesture detection script)

• requirements.txt

• README.md

Requirements
Install dependencies:
pip install opencv-python

pip install mediapipe

pip install streamlit

pip install pyttsx3 (optional for voice output)

Usage

Run the standalone detector:
python gesture.py
• Opens webcam

• Detects gestures

• Displays recognized phrase

Run the Streamlit interface:
python -m streamlit run app.py
• Shows live video preview

• Highlights detected gestures

• Lets you adjust thresholds and tuning

• Allows snapshot saving

• Supports optional text-to-speech

Technologies Used
Python, OpenCV, MediaPipe Hands, Streamlit, NumPy, pyttsx3 (optional)

About
A gesture-based assistive communication tool that helps users express essential phrases through simple hand signs, enabling smoother interaction in medical, educational, or accessibility-focused environments.
