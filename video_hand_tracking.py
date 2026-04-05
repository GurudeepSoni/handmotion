import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="Hand Tracking Video App", layout="wide")

# ==== HEADING ====
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🤚 Hand Tracking Video App 🤚</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a video and download hand-tracked result!</p>", unsafe_allow_html=True)

# ==== VIDEO UPLOAD ====
uploaded_file = st.file_uploader("🎥 Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Prepare MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    # Open uploaded video
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        
        out.write(frame)
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)
    
    cap.release()
    out.release()
    
    st.success("✅ Processing complete!")
    
    # Download button
    with open(output_file.name, "rb") as f:
        st.download_button("⬇️ Download Processed Video", f, file_name="hand_tracked_video.mp4")

st.markdown("<hr><p style='text-align: center;'>💖 Made by Akshita Soni 💖</p>", unsafe_allow_html=True)
