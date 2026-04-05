import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tempfile import NamedTemporaryFile
import tempfile
import time

st.set_page_config(page_title="Hand Tracking App", layout="wide")

# ==== AESTHETIC HEADING ====
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>✨🤚 Hand Tracking Magic App 🤚✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Real-time AI hand tracking made cool 😎🔥</p>", unsafe_allow_html=True)

# ==== NAVIGATION ====
mode = st.sidebar.radio("🎛️ Select Mode", ["Live Camera", "Video Upload"])

# ==== MEDIAPIPE SETUP ====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# ==== VIDEO PROCESSING FUNCTION ====
def process_frame(frame):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    return frame

# ==== LIVE CAMERA ====
if mode == "Live Camera":
    st.header("📷 Live Hand Tracking")
    run = st.checkbox("▶️ Start Camera")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        success, frame = cap.read()
        if not success:
            st.warning("Ignoring empty camera frame.")
            continue
        frame = process_frame(frame)
        FRAME_WINDOW.image(frame, channels="BGR")
    
    cap.release()

# ==== VIDEO UPLOAD ====
elif mode == "Video Upload":
    st.header("📂 Upload Video for Hand Tracking")
    uploaded_file = st.file_uploader("🎥 Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        FRAME_WINDOW = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            FRAME_WINDOW.image(frame, channels="BGR")
        
        cap.release()
        st.success("✅ Video processing complete!")

# ==== FOOTER ====
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>💖 Made by Akshita Soni 💖</p>", unsafe_allow_html=True)