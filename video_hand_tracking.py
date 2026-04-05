import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import cv2

st.set_page_config(page_title="Hand Tracking Video App", layout="wide")

# ==== AESTHETIC HEADING ====
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>✨🤚 Hand Tracking Video App 🤚✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Upload a video and get AI hand tracking results 😎🔥</p>", unsafe_allow_html=True)

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

# ==== VIDEO UPLOAD ====
st.header("📂 Upload Video for Hand Tracking")
uploaded_file = st.file_uploader("🎥 Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Video capture
    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Output video temp file
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

    stframe = st.empty()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        out.write(frame)
        stframe.image(frame, channels="BGR")
        current_frame += 1
        progress_bar.progress(min(current_frame / frame_count, 1.0))

    cap.release()
    out.release()
    st.success("✅ Video processing complete!")

    # Download button
    st.download_button(
        label="⬇️ Download Processed Video",
        data=open(output_file.name, 'rb').read(),
        file_name="hand_tracking_result.mp4",
        mime="video/mp4"
    )

# ==== FOOTER ====
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>💖 Made by Akshita Soni 💖</p>", unsafe_allow_html=True)
