Gesture Recognition Assistive Communication System using Python, OpenCV & MediaPipe

A real-time hand-gesture detection system that recognizes commonly used gestures and converts them into simple English phrases. Built to support individuals with speech impairments by providing a quick and intuitive communication method.

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

Screenshots-
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
