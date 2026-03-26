import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import joblib
import time
from collections import deque
from config import MODEL_PATH
import os

# ---------------------------
# Check model
# ---------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("🖐️ Real-Time Sign Language Recognition")
mode = st.radio("Select Mode", ["📷 Webcam", "🖼️ Upload Image"])
st.caption("Camera open → Sign detect → Meaning show → Sentence build → Clear button")
st.caption("Made By Mohammad Kaish")

# ---------------------------
# Load model
# ---------------------------
model = joblib.load(MODEL_PATH)

# ---------------------------
# MediaPipe setup (FIXED POSITION)
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ---------------------------
# SESSION STATE
# ---------------------------
if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "last_word" not in st.session_state:
    st.session_state.last_word = None

if "pred_buffer" not in st.session_state:
    st.session_state.pred_buffer = deque(maxlen=10)

if "running" not in st.session_state:
    st.session_state.running = False

if "last_added_time" not in st.session_state:
    st.session_state.last_added_time = 0.0

# ---------------------------
# LAYOUT
# ---------------------------
col1, col2 = st.columns([2, 1])

# ---------------------------
# UPLOAD MODE
# ---------------------------
if mode == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_word = "No Hand"

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)
            pred = model.predict(X)[0]
            current_word = pred

        st.image(frame, channels="BGR")
        st.success(f"Prediction: {current_word}")

# ---------------------------
# WEBCAM MODE
# ---------------------------
if mode == "📷 Webcam":

    with col2:
        st.subheader("🎮 Controls")

        if st.button("▶️ Start Camera"):
            st.session_state.running = True

        if st.button("⏹️ Stop Camera"):
            st.session_state.running = False

        if st.button("🧹 Clear Sentence"):
            st.session_state.sentence = []
            st.session_state.last_word = None

        st.divider()
        st.subheader("✅ Detected Word")
        detected_text = st.empty()

        st.subheader("🧾 Sentence")
        sentence_text = st.empty()

    with col1:
        st.subheader("📷 Live Camera")
        FRAME_WINDOW = st.image([])

    cap = None

    try:
        if st.session_state.running:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Cannot access camera")
                st.session_state.running = False

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Camera not working")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                current_word = "No Hand"

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    X = np.array(landmarks).reshape(1, -1)
                    pred = model.predict(X)[0]

                    st.session_state.pred_buffer.append(pred)
                    final_pred = max(set(st.session_state.pred_buffer),
                                     key=st.session_state.pred_buffer.count)

                    current_word = final_pred

                    now = time.time()
                    cooldown = 1.0

                    if (current_word != st.session_state.last_word) and (now - st.session_state.last_added_time > cooldown):
                        st.session_state.sentence.append(current_word)
                        st.session_state.last_word = current_word
                        st.session_state.last_added_time = now

                FRAME_WINDOW.image(frame, channels="BGR")
                detected_text.markdown(f"### **{current_word}**")
                sentence_text.markdown("### " + " ".join(st.session_state.sentence))

                time.sleep(0.03)

        else:
            detected_text.markdown("### **Stopped**")
            sentence_text.markdown("### " + " ".join(st.session_state.sentence))

    finally:
        if cap is not None:
            cap.release()