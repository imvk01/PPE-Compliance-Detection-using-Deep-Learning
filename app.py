from ultralytics import YOLO
import cv2
import os
from datetime import datetime

MODEL_PATH = "runs/detect/train-2/weights/best.pt"
VIDEO_PATH = "test_video.mp4"

model = YOLO(MODEL_PATH)

os.makedirs("outputs/imagesResult", exist_ok=True)
os.makedirs("outputs/videoResult", exist_ok=True)
os.makedirs("outputs/webcamResults", exist_ok=True)


def draw_boxes(frame, results):
    detected_classes = []

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
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + 240, y1), color, -1)

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    return frame, detected_classes


def image_detection():
    image_folder = "test_images"

    if not os.path.exists(image_folder):
        print("\nFolder test_images not found.")
        return

    image_files = os.listdir(image_folder)

    if len(image_files) == 0:
        print("\nNo images found inside test_images folder.")
        return

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        results = model.predict(frame, conf=0.4, verbose=False)

        frame, detected_classes = draw_boxes(frame, results)

        save_path = os.path.join("outputs/imagesResult", image_name)
        cv2.imwrite(save_path, frame)

    print("\nImage detection completed.")
    print("Results saved in outputs/imagesResult")

def video_detection():

    video_folder = "test_videos"

    if not os.path.exists(video_folder):
        print("\nFolder test_videos not found.")
        return

    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv")

    video_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith(allowed_extensions)
    ]

    if len(video_files) == 0:
        print("\nNo valid videos found inside test_videos folder.")
        return

    for video_name in video_files:

        video_path = os.path.join(video_folder, video_name)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"\nCould not open video: {video_name}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            fps = 25

        output_path = os.path.join(
            "outputs/videoResult",
            f"detected_{video_name}"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(frame, conf=0.4, verbose=False)

            frame, detected_classes = draw_boxes(frame, results)

            out.write(frame)

        cap.release()
        out.release()

        print(f"\nProcessed video: {video_name}")

    print("\nAll video detections completed.")
    print("Results saved in outputs/videoResult")

    
def webcam_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\nCamera could not be opened.")
        return

    

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(frame, conf=0.4, verbose=False)

        frame, detected_classes = draw_boxes(frame, results)

        helmet_detected = "helmet" in detected_classes
        head_detected = "head" in detected_classes

        if head_detected and not helmet_detected:
            status_text = "WARNING: NO HELMET DETECTED"
            status_color = (0, 0, 255)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"outputs/webcamResults/violation_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)

        elif helmet_detected:
            status_text = "PPE COMPLIANT"
            status_color = (0, 255, 0)

        else:
            status_text = "NO PERSON/HEAD DETECTED"
            status_color = (0, 255, 255)

        cv2.putText(
            frame,
            status_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            3
        )

        cv2.imshow("PPE Compliance Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nWebcam detection stopped.")
    print("Violation images saved in outputs/webcamResults")


while True:
    print("\n========== PPE COMPLIANCE DETECTION ==========")
    print("1. Image Detection")
    print("2. Video Detection")
    print("3. Webcam Detection")
    print("4. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        image_detection()

    elif choice == "2":
        video_detection()

    elif choice == "3":
        webcam_detection()

    elif choice == "4":
        print("\nExiting program...")
        break

    else:
        print("\nInvalid choice. Please try again.")