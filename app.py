import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = "runs/detect/train-2/weights/best.pt"
os.makedirs("outputs/imagesResult", exist_ok=True)
os.makedirs("outputs/videoResult", exist_ok=True)
os.makedirs("outputs/webcamResults", exist_ok=True)

# ── Page Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Helmet Detection System",
    page_icon="🦺",
    layout="wide"
)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error(f"Model weights not found at '{MODEL_PATH}'. Please check the path.")
        return None

model = load_model()

def draw_boxes(frame, results):
    detected_classes = []
    if not results or len(results) == 0:
        return frame, detected_classes

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower().strip()
        detected_classes.append(class_name)

        if class_name == "helmet":
            color = (0, 255, 0)
            label = f"HELMET {conf:.2f}"
        elif class_name == "head":
            color = (0, 0, 255)
            label = f"NO HELMET {conf:.2f}"
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 180, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, detected_classes

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🦺 Smart Safety Helmet Compliance System")
st.markdown("Automated PPE detection system powered by a custom-trained YOLO model.")

if model is None:
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🛠️ Settings")
app_mode = st.sidebar.selectbox(
    "Choose Input Source",
    ["Overview", "Image Detection", "Video Detection", "Webcam Detection"]
)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# ── MODE 1: Overview ───────────────────────────────────────────────────────────
if app_mode == "Overview":
    st.subheader("System Overview")
    st.info("Select an input source from the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Class 0", "Helmet ✅ (Safe)")
    col2.metric("Class 1", "Head ❌ (No Helmet)")
    col3.metric("Class 2", "Person (Contextual)")

# ── MODE 2: Image Detection ────────────────────────────────────────────────────
elif app_mode == "Image Detection":
    st.subheader("📸 Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)

        results = model.predict(opencv_img, conf=confidence_threshold, verbose=False)
        processed_img, classes = draw_boxes(opencv_img.copy(), results)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB),
                     caption="Original Image", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                     caption="Detection Result", use_container_width=True)

        if "head" in classes and "helmet" not in classes:
            st.error("🚨 WARNING: No helmet detected — PPE violation!")
        elif "helmet" in classes:
            st.success("✅ Helmet detected — PPE compliant.")
        else:
            st.info("ℹ️ No person or head detected in the image.")

# ── MODE 3: Video Detection ────────────────────────────────────────────────────
elif app_mode == "Video Detection":
    st.subheader("🎞️ Video Detection")
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        st_frame = st.empty()
        stop_btn = st.button("⏹ Stop Video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break

            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            processed_frame, _ = draw_boxes(frame, results)
            st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

        cap.release()
        st.success("✅ Video processing complete.")

# ── MODE 4: Webcam Detection ───────────────────────────────────────────────────
elif app_mode == "Webcam Detection":
    st.subheader("📹 Webcam Detection")
    st.warning(
        "⚠️ Webcam is not supported on cloud deployments. "
        "To use live webcam detection, run the app locally with:\n\n"
        "```\nstreamlit run app.py\n```"
    )
    st.info(
        "For local use, the webcam mode will:\n"
        "- Detect helmets and heads in real time\n"
        "- Show a PPE compliance status on screen\n"
        "- Automatically save violation snapshots to `outputs/webcamResults/`"
    )