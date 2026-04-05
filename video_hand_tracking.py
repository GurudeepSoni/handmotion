import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="Hand Tracking Video App", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🤚 Hand Tracking Video App 🤚</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Upload video and track hand movements</p>", unsafe_allow_html=True)

# ==== NAVIGATION ====
st.header("📂 Upload Video for Hand Tracking")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(tfile.name)

    # Temp file for output video
    output_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_tempfile.name, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        
        out.write(frame)
        processed_frames += 1
        progress_bar.progress(min(processed_frames / total_frames, 1.0))
    
    cap.release()
    out.release()

    st.success("✅ Video processing complete!")

    # Download button
    with open(output_tempfile.name, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f,
            file_name="hand_tracked_video.mp4",
            mime="video/mp4"
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>💖 Made by Akshita Soni 💖</p>", unsafe_allow_html=True)
