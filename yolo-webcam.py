from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../yolo-weights/yolov8n.pt")


ClassNames = model.names

while True:
    success, img = cap.read()
    if not success:
        break

    
    results = model(img, stream=True)

  
    img = cv2.flip(img, 1)

    # Bounding Box
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            
            img_width = img.shape[1]
            x1, x2 = img_width - x2, img_width - x1

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