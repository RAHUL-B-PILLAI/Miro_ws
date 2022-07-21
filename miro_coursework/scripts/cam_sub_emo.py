#!/usr/bin/env python3
from __future__ import print_function

#import roslib
#roslib.load_manifest('unibas_face_detector')
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from deepface import DeepFace

class viewer:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.callback)
    self.pub = rospy.Publisher('/face_detector/faces', Image, queue_size=10)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)    

    #cv2.imshow("faces", cv_image)
    face_cascade = cv2.CascadeClassifier('/home/rahul/ros_workspace/miro_ws/src/miro_coursework/data/haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    font= cv2.FONT_HERSHEY_SIMPLEX
    for (x,y,w,h) in faces:
        cv2.rectangle(cv_image,(x,y),(x+w,y+h),(255,0,0),2)
        prediction=DeepFace.analyze(cv_image, actions=['emotion'],enforce_detection=True)
        cv2.putText(cv_image,prediction['dominant_emotion'],(x,y),font,3,(0,0,255),2,cv2.LINE_4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = cv_image[y:y+h, x:x+w]
    cv2.imshow("faces", cv_image)
    cv2.waitKey(2)

    faces_message = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
    self.pub.publish(faces_message)
   

def main():
  v = viewer()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('viewer_node')
    main()