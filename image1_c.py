#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

from ObjectDetection import ObjectDetection


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    # iterator to capture images
    self.iterator = 0
    #
    self.od = ObjectDetection()
    # targets
    self.targets_pub = rospy.Publisher("targets_pos",Float64MultiArray, queue_size=10)
    # initialize target array
    self.targets = Float64MultiArray()

    self.cv_image1 = cv2.imread("image_1_1.png", 1)


  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    self.iterator += 1
    # Recieve the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #if self.iterator % 50 == 0:
    #    print(self.iterator)
    #    cv2.imwrite('image_1_'+str(self.iterator/50)+'.png', self.cv_image1)
    #im1=cv2.imshow('window1', self.cv_image1)
    #cv2.waitKey(1)

    img = self.od.filter_colour(self.cv_image1, "orange")
    img = self.od.opening(img, kernel_size=3)
    img = self.od.dilate(img, 3)
    boundries, contours = self.od.find_boundries(img)

    try:
        cx0, cy0 = self.od.get_center(img, boundries[0])
        cx1, cy1 = self.od.get_center(img, boundries[1])
        #obj = od.get_object(img2, rects[j])
        a = np.array([cx0, cy0, cx1, cy1])
        self.targets.data = a
    except:
        self.targets.data = np.array([0, 0, 0, 0])

    #rects, cnts = od.find_boundries(img2)

    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      self.targets_pub.publish(self.targets)
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
