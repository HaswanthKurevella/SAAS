import cv2
import numpy as np
import dlib


cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()

while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()