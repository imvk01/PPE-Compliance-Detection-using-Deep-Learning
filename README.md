🦺 PPE Compliance Detection using Deep Learning
📌 Overview

This project is an AI-based PPE (Personal Protective Equipment) Compliance Detection System using YOLOv8 and OpenCV.

The system detects:

1. Helmet
2. No Helmet
3. Head

from:

Images
Videos
Real-time Webcam Feed

If a worker is detected without a helmet, the system displays a warning and saves violation screenshots automatically.

🚀 Features
Real-time Webcam Detection
Image Detection
Video Detection
Helmet / No Helmet Detection
Automatic Violation Screenshot Saving
Bounding Box Visualization
Confidence Score Display
Multiple Image & Video Support
🧠 Technologies Used
Python
YOLOv8
OpenCV
PyTorch
Ultralytics
📂 Project Structure
PPE-Compliance-Detection/
│
├── dataset/
├── outputs/
├── runs/
├── test_images/
├── test_videos/
│
├── app.py
├── train.py
├── detect_image.py
├── detect_video.py
├── webcam_detection.py
├── requirements.txt
└── data.yaml


⚙️ Installation
1️⃣ Clone Repository
git clone <your-repository-link>
cd PPE-Compliance-Detection
2️⃣ Install Requirements
Mac/Linux
pip3 install -r requirements.txt
Windows
pip install -r requirements.txt
🏋️‍♂️ Train Model
python3 train.py

Best model saves in:

runs/detect/train-2/weights/best.pt
🖼️ Image Detection

Put images inside:

test_images/

Run:

python3 app.py

Choose:

1 → Image Detection

Results save in:

outputs/imagesResult
🎥 Video Detection

Put videos inside:

test_videos/

Run:

python3 app.py

Choose:

2 → Video Detection

Results save in:

outputs/videoResult
📹 Webcam Detection

Run:

streamlit run app.py

Choose:

3 → Webcam Detection

Press:

ctrl/cmd + c

to exit.

Violation screenshots save in:

outputs/webcamResults
⚠️ PPE Logic
Detection	Result
Helmet detected	PPE COMPLIANT
Head without helmet	WARNING: NO HELMET DETECTED
📊 Dataset

Dataset Used: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection?utm_source=chatgpt.com

👨‍💻 Author

Vikash Verma

Live: https://ppe-12.streamlit.app/

GitHub: https://github.com/imvk01/PPE-Compliance-Detection-using-Deep-Learning

LinkedIn: https://www.linkedin.com/in/imvk1/
