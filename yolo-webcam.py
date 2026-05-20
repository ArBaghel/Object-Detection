from ultralytics import YOLO
import cv2
import cvzone
import math
import pickle
import os

# --- 1. INITIALIZE CAPTURE AND LOAD MODEL ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../yolo-weights/yolov8n.pt")
ClassNames = model.names

# --- 2. THE WEBCAM DETECTION LOOP ---
while True:
    success, img = cap.read()
    if not success:
        break

    # 🔥 FIX: Flip the image FIRST so YOLO reads the mirrored frame directly
    img = cv2.flip(img, 1)
    
    # Send the already-flipped image to the model
    results = model(img, stream=True)

    # Bounding Box Logic
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 🔥 FIX: Manual subtraction math removed! 
            # Because YOLO looked at the flipped image, x1 and y1 are already perfectly positioned.
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{ClassNames[cls]} {conf}',
                               (max(0, x1), max(35, y1)),
                               scale=0.6, thickness=1, offset=3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print("\n📦 Webcam window closed. Starting pipeline serialization...")

exported_weights_name = "app_model_weights.pt"
model.save(exported_weights_name)
print(f"✔️ Saved model weights structure to: '{exported_weights_name}'")

metadata = {
    "class_names": ClassNames,
    "default_width": 1280,
    "default_height": 720,
    "model_version": "YOLOv8n-Production"
}

pickle_filename = "model_config.pkl"
with open(pickle_filename, "wb") as pickle_file:
    pickle.dump(metadata, pickle_file)

print(f"✔️ Successfully generated PICKLE file configuration: '{pickle_filename}'")