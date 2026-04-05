import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import tempfile

st.set_page_config(page_title="Hand Tracking App", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>✨🤚 Hand Tracking Magic App 🤚✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Real-time AI hand tracking 😎🔥</p>", unsafe_allow_html=True)

mode = st.sidebar.radio("🎛️ Select Mode", ["Live Camera", "Video Upload"])

# ==== MEDIAPIPE SETUP ====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandTracker(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
        return img

# ==== LIVE CAMERA USING WEBRTC ====
if mode == "Live Camera":
    st.header("📷 Live Hand Tracking")
    st.warning("🔔 Browser will ask for camera permission")
    webrtc_streamer(
        key="hand-tracking",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=HandTracker,
        media_stream_constraints={"video": True, "audio": False}
    )

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
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7
            ).process(imgRGB)
            
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            FRAME_WINDOW.image(frame, channels="BGR")
        cap.release()
        st.success("✅ Video processing complete!")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>💖 Made by Akshita Soni 💖</p>", unsafe_allow_html=True)
