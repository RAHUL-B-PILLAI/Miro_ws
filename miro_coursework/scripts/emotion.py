import cv2
from deepface import DeepFace

IMG=cv2.imread('original_photo_woman_happy-1.png')

#prediction=DeepFace.analyze(IMG, actions=['emotion'])

#print(prediction)

face_cascade= cv2.CascadeClassifier('/home/rahul/ros_workspace/miro_ws/src/miro_coursework/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame= cap.read()
    font= cv2.FONT_HERSHEY_SIMPLEX
    
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        color = (255,0,0)
        stroke=2
        prediction=DeepFace.analyze(frame, actions=['emotion'])
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,prediction['dominant_emotion'],(x,y),font,3,(0,0,255),2,cv2.LINE_4)
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
