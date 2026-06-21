# from ultralytics import YOLO
# import cv2
# import os
# from datetime import datetime

# MODEL_PATH = "runs/detect/train-2/weights/best.pt"
# VIDEO_PATH = "test_video.mp4"

# model = YOLO(MODEL_PATH)

# os.makedirs("outputs/imagesResult", exist_ok=True)
# os.makedirs("outputs/videoResult", exist_ok=True)
# os.makedirs("outputs/webcamResults", exist_ok=True)


# def draw_boxes(frame, results):
#     detected_classes = []

#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         class_id = int(box.cls[0])
#         class_name = model.names[class_id].lower().strip()

#         detected_classes.append(class_name)

#         if class_name == "helmet":
#             color = (0, 255, 0)
#             label = f"HELMET {conf:.2f}"

#         elif class_name == "head":
#             color = (0, 0, 255)
#             label = f"NO HELMET {conf:.2f}"

#         else:
#             continue

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(frame, (x1, y1 - 35), (x1 + 240, y1), color, -1)

#         cv2.putText(
#             frame,
#             label,
#             (x1 + 5, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (255, 255, 255),
#             2
#         )

#     return frame, detected_classes


# def image_detection():
#     image_folder = "test_images"

#     if not os.path.exists(image_folder):
#         print("\nFolder test_images not found.")
#         return

#     image_files = os.listdir(image_folder)

#     if len(image_files) == 0:
#         print("\nNo images found inside test_images folder.")
#         return

#     for image_name in image_files:
#         image_path = os.path.join(image_folder, image_name)
#         frame = cv2.imread(image_path)

#         if frame is None:
#             continue

#         results = model.predict(frame, conf=0.4, verbose=False)

#         frame, detected_classes = draw_boxes(frame, results)

#         save_path = os.path.join("outputs/imagesResult", image_name)
#         cv2.imwrite(save_path, frame)

#     print("\nImage detection completed.")
#     print("Results saved in outputs/imagesResult")

# def video_detection():

#     video_folder = "test_videos"

#     if not os.path.exists(video_folder):
#         print("\nFolder test_videos not found.")
#         return

#     allowed_extensions = (".mp4", ".avi", ".mov", ".mkv")

#     video_files = [
#         f for f in os.listdir(video_folder)
#         if f.lower().endswith(allowed_extensions)
#     ]

#     if len(video_files) == 0:
#         print("\nNo valid videos found inside test_videos folder.")
#         return

#     for video_name in video_files:

#         video_path = os.path.join(video_folder, video_name)

#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             print(f"\nCould not open video: {video_name}")
#             continue

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))

#         if fps == 0:
#             fps = 25

#         output_path = os.path.join(
#             "outputs/videoResult",
#             f"detected_{video_name}"
#         )

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         while cap.isOpened():

#             ret, frame = cap.read()

#             if not ret:
#                 break

#             results = model.predict(frame, conf=0.4, verbose=False)

#             frame, detected_classes = draw_boxes(frame, results)

#             out.write(frame)

#         cap.release()
#         out.release()

#         print(f"\nProcessed video: {video_name}")

#     print("\nAll video detections completed.")
#     print("Results saved in outputs/videoResult")

    
# def webcam_detection():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("\nCamera could not be opened.")
#         return

    

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         results = model.predict(frame, conf=0.4, verbose=False)

#         frame, detected_classes = draw_boxes(frame, results)

#         helmet_detected = "helmet" in detected_classes
#         head_detected = "head" in detected_classes

#         if head_detected and not helmet_detected:
#             status_text = "WARNING: NO HELMET DETECTED"
#             status_color = (0, 0, 255)

#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             save_path = f"outputs/webcamResults/violation_{timestamp}.jpg"
#             cv2.imwrite(save_path, frame)

#         elif helmet_detected:
#             status_text = "PPE COMPLIANT"
#             status_color = (0, 255, 0)

#         else:
#             status_text = "NO PERSON/HEAD DETECTED"
#             status_color = (0, 255, 255)

#         cv2.putText(
#             frame,
#             status_text,
#             (20, 45),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             status_color,
#             3
#         )

#         cv2.imshow("PPE Compliance Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     print("\nWebcam detection stopped.")
#     print("Violation images saved in outputs/webcamResults")


# while True:
#     print("\n========== PPE COMPLIANCE DETECTION ==========")
#     print("1. Image Detection")
#     print("2. Video Detection")
#     print("3. Webcam Detection")
#     print("4. Exit")

#     choice = input("\nEnter your choice: ")

#     if choice == "1":
#         image_detection()

#     elif choice == "2":
#         video_detection()

#     elif choice == "3":
#         webcam_detection()

#     elif choice == "4":
#         print("\nExiting program...")
#         break

#     else:
#         print("\nInvalid choice. Please try again.")

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# Configuration & Directories Setup
MODEL_PATH = "runs/detect/train-2/weights/best.pt"
os.makedirs("outputs/imagesResult", exist_ok=True)
os.makedirs("outputs/videoResult", exist_ok=True)
os.makedirs("outputs/webcamResults", exist_ok=True)

# Page Configuration
st.set_page_config(page_title="PPE Helmet Detection System", page_icon="🦺", layout="wide")

@st.cache_resource
def load_model():
    """Cache the model initialization so it doesn't reload on every interaction."""
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error(f"Model weights not found at '{MODEL_PATH}'. Please verify your weights location.")
        return None

model = load_model()

def draw_boxes(frame, results):
    """Draws bounding boxes and computes status arrays based on YOLO predictions."""
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
            color = (0, 255, 0) # Green
            label = f"HELMET {conf:.2f}"
        elif class_name == "head":
            color = (0, 0, 255) # Red
            label = f"NO HELMET {conf:.2f}"
        else:
            continue

        # Draw bounding boxes and text backgrounds
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 180, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, detected_classes

# --- FRONTEND INTERFACE DESIGN ---
st.title("🦺 Smart Safety Helmet Compliance System")
st.markdown("Automated PPE object monitoring system utilizing a customized YOLO neural architecture.")

if model is None:
    st.stop()

# Sidebar Control Center
st.sidebar.header("🛠️ Control Settings")
app_mode = st.sidebar.selectbox("Choose Input Source", ["Overview & Statistics", "Photo Upload Mode", "Video Upload Mode", "Live Webcam Stream"])
confidence_threshold = st.sidebar.slider("Model Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# --- MODE 1: OVERVIEW & INSTRUCTIONS ---
if app_mode == "Overview & Statistics":
    st.subheader("System Overview")
    st.info("Select an operations modality from the sidebar dropdown control vector to initiate asset safety processing inference.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Label Index 0", "Helmet (Safe)")
    col2.metric("Target Label Index 1", "Head (Unsafe)")
    col3.metric("Target Label Index 2", "Person (Contextual)")

# --- MODE 2: PHOTO PROCESSING ---
elif app_mode == "Photo Upload Mode":
    st.subheader("📸 Digital Target Image Frame Parsing")
    uploaded_file = st.file_uploader("Upload target format target frame...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_img = cv2.imdecode(file_bytes, 1)
        
        # Inference pass
        results = model.predict(opencv_img, conf=confidence_threshold, verbose=False)
        processed_img, classes = draw_boxes(opencv_img.copy(), results)
        
        # Display structures side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), caption="Original Asset Frame Input", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Evaluated Frame Structure Tensor Output", use_container_width=True)
            
            # Analytics feedback loop
            if "head" in classes:
                st.error("🚨 Critical Alert Condition triggered: Non-compliant worker context evaluated.")
            elif "helmet" in classes:
                st.success("✅ Frame Environment Checked: Standard operational safety metrics verified.")

# --- MODE 3: VIDEO FRAME RENDERING ---
elif app_mode == "Video Upload Mode":
    st.subheader("🎞️ Pre-recorded Continuous Video Sequence Stream Processing")
    uploaded_video = st.file_uploader("Upload structural work video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        tfile = open("temp_target_stream.mp4", "wb")
        tfile.write(uploaded_video.read())
        tfile.close()
        
        cap = cv2.VideoCapture("temp_target_stream.mp4")
        st_frame = st.empty()
        
        stop_btn = st.button("Halt Video Rendering Process Loop")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
                
            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            processed_frame, _ = draw_boxes(frame, results)
            
            # Send continuous stream frames directly to UI window block
            st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video data vector execution terminal sequence complete.")

# --- MODE 4: LIVE WEBCAM PROCESSING ---
elif app_mode == "Live Webcam Stream":
    st.subheader("📹 Real-time Device Local Frame Stream Analysis")
    st.warning("Ensure webcam device execution pipelines are clear of lock conflicts by background programs.")
    
    run_webcam = st.checkbox("Initialize Webcam Interface Driver Loop", value=False)
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        st_webcam_frame = st.empty()
        st_status_box = st.empty()
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Peripheral execution camera acquisition channel dropped stream array packet.")
                break
                
            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            processed_frame, classes = draw_boxes(frame, results)
            
            # Status alert metrics mapping logic loop
            if "head" in classes and "helmet" not in classes:
                st_status_box.error("⚠️ ALERT VECTOR GENERATED: Operational field violation identified - No Helmet!")
                # Disk violation caching engine
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"outputs/webcamResults/violation_{timestamp}.jpg", processed_frame)
            elif "helmet" in classes:
                st_status_box.success("🦺 Compliance Metrics: Clear target confirmation.")
            else:
                st_status_box.info("🔍 Active Search Pattern: No actionable safety targets matched.")
                
            st_webcam_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
        cap.release()