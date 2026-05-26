from ultralytics import YOLO
import cv2
import os

# =========================
# LOAD MODEL
# =========================
model = YOLO("runs/detect/train-2/weights/best.pt")

# =========================
# CREATE OUTPUT FOLDER
# =========================
os.makedirs("outputs/videoResult", exist_ok=True)

# =========================
# VIDEO PATH
# =========================
video_path = "test.mp4"

# =========================
# OPEN VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video could not be opened.")
    exit()

# =========================
# VIDEO PROPERTIES
# =========================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# =========================
# OUTPUT VIDEO
# =========================
output_path = "outputs/videoResult/output_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Processing video...")

# =========================
# PROCESS VIDEO
# =========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # PREDICTION
    # =========================
    results = model.predict(
        frame,
        conf=0.4,
        verbose=False
    )

    # =========================
    # DRAW DETECTIONS
    # =========================
    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        class_id = int(box.cls[0])

        class_name = model.names[class_id].lower().strip()

        # =========================
        # HELMET
        # =========================
        if class_name == "helmet":

            color = (0, 255, 0)
            label = f"HELMET {conf:.2f}"

        # =========================
        # NO HELMET
        # =========================
        elif class_name == "head":

            color = (0, 0, 255)
            label = f"NO HELMET {conf:.2f}"

        else:
            continue

        # =========================
        # DRAW BOX
        # =========================
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # =========================
        # LABEL BACKGROUND
        # =========================
        cv2.rectangle(
            frame,
            (x1, y1 - 35),
            (x1 + 240, y1),
            color,
            -1
        )

        # =========================
        # LABEL TEXT
        # =========================
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # =========================
    # SAVE FRAME
    # =========================
    out.write(frame)

    # =========================
    # SHOW VIDEO
    # =========================
    cv2.imshow("Video Detection", frame)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# RELEASE
# =========================
cap.release()
out.release()

cv2.destroyAllWindows()

print("Video detection completed.")
print("Saved in outputs/videoResult/")