#  YOLOv8 Object Detection

This project demonstrates **object detection using YOLOv8**, showcasing
real-time inference, bounding-box predictions, and practical
computer-vision workflows. Built as a hands-on exploration of modern
detection models and CV fundamentals.

##  Overview

This repository covers:

-   Object detection using **Ultralytics YOLOv8**
-   Inference on images & videos
-   Training on custom datasets
-   Clean, modular scripts for easy adaptation

##  Project Structure

    /
    ├── data/                  # sample images/videos (optional)
    ├── models/                # pretrained/custom model weights
    ├── src/
    │   ├── detect.py          # inference script
    │   ├── train.py           # training script
    │   └── utils.py           # helper functions
    ├── requirements.txt       # pip dependencies
    └── README.md

##  Installation

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/ArBaghel/Object-Detection
cd Object-Detection
```

### 2️⃣ Create a virtual environment (optional)

``` bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
.env\Scriptsctivate         # Windows
```

### 3️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

##  Run Object Detection

Run YOLOv8 on any image or video:

``` bash
python src/detect.py --source path/to/image_or_video --weights yolov8n.pt
```

Outputs will be saved automatically with bounding boxes and class
labels.

##  Train on a Custom Dataset

1.  Prepare dataset in **YOLO format** (images + label `.txt` files)
2.  Configure your `.yaml` dataset file
3.  Run training:

``` bash
python src/train.py --data path/to/dataset.yaml --epochs 50
```

After training, use the generated `best.pt` weights for detection.

##  Features

-    Real-time object detection,
-    Webcam live detection,
-    Works with custom datasets,
-    Extendable code structure,
-    Fast inference via YOLOv8,
-    Supports images, videos, camera streams.

##  Why YOLOv8?

-   Cutting-edge YOLO architecture,
-   Highly optimized for speed + accuracy,
-   Simple API with Ultralytics,
-   Supports detection, segmentation, classification, pose estimation.

Perfect for projects, demos, prototyping, and real-world ML
applications.

##  Future Enhancements

-   Segmentation + pose estimation support,
-   Performance metrics (mAP, F1, confusion matrix),
-   Config-driven training & inference.

##  Contribute

Feel free to open issues, contribute improvements, or suggest new
features.
If you like this project --- **⭐ Star the Repo!**
