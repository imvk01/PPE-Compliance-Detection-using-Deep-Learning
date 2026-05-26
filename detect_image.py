from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("runs/detect/train-2/weights/best.pt")

image_folder = "test_images"

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)

    results = model(image)[0]

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        cls = int(box.cls[0])

        class_name = model.names[cls]

        # --------------------------------
        # HELMET CLASS
        # --------------------------------
        if class_name.lower() == "helmet":

            color = (0, 255, 0)  # Green
            label = f"HELMET {conf:.2f}"

        # --------------------------------
        # NO HELMET / HEAD
        # --------------------------------
        elif class_name.lower() == "head":

            color = (0, 0, 255)  # Red
            label = f"NO HELMET {conf:.2f}"

        else:
            continue

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # Label background
        cv2.rectangle(
            image,
            (x1, y1 - 35),
            (x1 + 220, y1),
            color,
            -1
        )

        # Put text
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    save_path = os.path.join(output_folder, image_name)

    cv2.imwrite(save_path, image)

print("Detection completed.")
print("Results saved in outputs/")