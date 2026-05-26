from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# =========================
# LOAD MODEL
# =========================
model = YOLO("runs/detect/train-2/weights/best.pt")

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs("outputs/webcamResults", exist_ok=True)

# =========================
# START WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Press Q to quit...")

# =========================
# WEBCAM LOOP
# =========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # YOLO PREDICTION
    # =========================
    results = model.predict(
        frame,
        conf=0.4,
        verbose=False
    )

    # Copy frame
    annotated_frame = frame.copy()

    detected_classes = []

    # =========================
    # PROCESS DETECTIONS
    # =========================
    for box in results[0].boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confidence
        conf = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Class name
        class_name = model.names[class_id].lower().strip()

        detected_classes.append(class_name)

        # =========================
        # HELMET DETECTED
        # =========================
        if class_name == "helmet":

            color = (0, 255, 0)  # Green
            label = f"HELMET {conf:.2f}"

        # =========================
        # NO HELMET / HEAD
        # =========================
        elif class_name == "head":

            color = (0, 0, 255)  # Red
            label = f"NO HELMET {conf:.2f}"

        else:
            continue

        # =========================
        # DRAW RECTANGLE
        # =========================
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # =========================
        # LABEL BACKGROUND
        # =========================
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - 35),
            (x1 + 240, y1),
            color,
            -1
        )

        # =========================
        # LABEL TEXT
        # =========================
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # =========================
    # STATUS CHECK
    # =========================
    helmet_detected = "helmet" in detected_classes
    head_detected = "head" in detected_classes

    if head_detected and not helmet_detected:

        status_text = "WARNING: NO HELMET DETECTED"
        status_color = (0, 0, 255)

        # Save violation image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = (
            f"outputs/webcamResults/"
            f"violation_{timestamp}.jpg"
        )

        cv2.imwrite(save_path, annotated_frame)

    elif helmet_detected:

        status_text = "PPE COMPLIANT"
        status_color = (0, 255, 0)

    else:

        status_text = "NO PERSON/HEAD DETECTED"
        status_color = (0, 255, 255)

    # =========================
    # SHOW STATUS TEXT
    # =========================
    cv2.putText(
        annotated_frame,
        status_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3
    )

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow(
        "PPE Compliance Detection",
        annotated_frame
    )

    # =========================
    # PRESS Q TO EXIT
    # =========================
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# RELEASE RESOURCES
# =========================
cap.release()
cv2.destroyAllWindows()

print("Webcam detection stopped.")
print("Violation images saved in outputs/webcamResults/")