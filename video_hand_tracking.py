import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import cv2

st.set_page_config(page_title="Hand Movement Tracker", layout="wide")

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>🤚 Hand Movement Tracker 🤚</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload your video and get hand-tracked result!</p>", unsafe_allow_html=True)

# ==== Video Upload ====
uploaded_file = st.file_uploader("🎥 Upload Video (mp4/mov/avi)", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.7)

    # Output video setup
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    stframe = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        out.write(frame)

        # Show in Streamlit
        stframe.image(frame, channels="BGR")
        processed_frames += 1
        progress_bar.progress(processed_frames / total_frames)

    cap.release()
    out.release()
    hands.close()

    st.success("✅ Video processing complete!")
    st.video(output_file)

    with open(output_file, "rb") as f:
        st.download_button("⬇️ Download Result Video", f, file_name="hand_tracked_video.mp4")
