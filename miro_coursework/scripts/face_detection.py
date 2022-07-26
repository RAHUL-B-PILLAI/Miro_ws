#!/usr/bin/env python3

import cv2
import numpy as numpy


face_cascade= cv2.CascadeClassifier('/home/rahul/ros_workspace/miro_ws/src/miro_coursework/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame= cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        color = (255,0,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
