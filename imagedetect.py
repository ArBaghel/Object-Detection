from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
model=YOLO('yolov8n.pt')
result=model('img/bus.jpg',show=True)

result[0].show()
