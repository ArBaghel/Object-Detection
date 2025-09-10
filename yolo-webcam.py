from ultralytics import YOLO
import cv2
import cvzone
import math
 


#cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture(0)
model=YOLO("../yolo-weights/yolov8n.pt")
ClassNames = ["person", "bicycle", "car",
              "motorbike", "aeroplane", "bus","train", "truck", "boat","ambulance","fire-brigade"
              "traffic light", "fire hydrant","stop sign", "parking meter",
              "bench", "bird", "cat", "dog","horse", "sheep", "cow",
              "elephant", "bear", "zebra","giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis",
              "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
              "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush","earphone"
              ]
while True:
    success, img = cap.read()
    results=model(img,stream=True)

    img=cv2.flip(img,1)

    #Bounding Box
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1, y1, x2, y2=box.xyxy[0]
            x1, y1, x2, y2= int(x1), int(y1), int(x2), int(y2)
            img_width = img.shape[1]  # Get image width
            x1, x2 = img_width - x2, img_width - x1  # Adjust coordinates after flipping
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
             #confidence
            conf=math.ceil((box.conf[0]*100))/100
            print(conf)
            #Class name
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{ClassNames[cls]}{conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
