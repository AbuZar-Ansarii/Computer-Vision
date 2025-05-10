import torch
import cv2
from sympy import resultant

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

url = 'http://192.168.31.93:8080/video'  # used mobile cam by ip webcam app

cap = cv2.VideoCapture(url) # replace url by 0 if using laptop cam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)
    anoted_frame = result.render()[0]
    cv2.imshow('YOLOv5 Detection',anoted_frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

