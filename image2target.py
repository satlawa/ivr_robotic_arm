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
import message_filters

from ObjectDetection import ObjectDetection
from SvmClassifier import classifer


class image2target:


    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joints_pub = rospy.Publisher("joints_pos",Float64MultiArray, queue_size=10)
        self.target_pub = rospy.Publisher("target_pos",Float64MultiArray, queue_size=10)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = message_filters.Subscriber("/image_topic1",Image)
        self.image_sub2 = message_filters.Subscriber("/image_topic2",Image)

        sync = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 10)
        sync.registerCallback(self.callback_sync)

        #rospy.TimeSynchronizer
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # iterator to capture images
        self.iterator = 0

        # use class for object detection
        self.od = ObjectDetection()
        self.svm = classifer([['obj0', 8], ['obj1', 8]])
        self.svm.addTrainSamples('images_train')
        self.svm.train('svm.xml')

    def detect_joints(self, img):
        joints = []
        for colour in ["red", "green", "blue", "yellow"]:
            mask = self.od.filter_color(img, colour)
            mask = self.od.dilate(img=mask, kernel_size=5)
            cx, cy = self.od.get_center_joint(mask)
            joints.append([cx, cy])
        return joints

    def detect_target(self, img):
        img = self.od.filter_colour(img, "orange")
        img = self.od.opening(img, kernel_size=3)
        img = self.od.dilate(img, 3)
        boundries, contours = self.od.find_boundries(img)
        return img, boundries, contours


    # Recieve data from camera 1, process it, and publish
    def callback_sync(self, data0, data1):
        self.iterator += 1
        # Recieve the image
        try:
            self.cv_image0 = self.bridge.imgmsg_to_cv2(data0, "bgr8")
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Uncomment if you want to save the image
        #if self.iterator % 50 == 0:
        #    print(self.iterator)
        #    cv2.imwrite('image_1_'+str(self.iterator/50)+'.png', self.cv_image1)
        # show images
        #im1=cv2.imshow('window1', self.cv_image0)
        #im2=cv2.imshow('window2', self.cv_image1)
        #cv2.waitKey(1)

        img, boundries, contours = self.detect_target(self.cv_image0)

        try:
            # get center
            cx0, cy0 = self.od.get_center_target(img, boundries[0])
            obj0 = self.od.get_object(img, boundries[0])
            prediction0 = self.svm.classify(obj0)
            data0 = [int(prediction0[1][0]), cx0, cy0] #prediction1[1][0],
            #print(prediction, cx, cy)
        except:
            #self.targets.data = np.array([0, 0, 0, 0])
            data0 = [0, 0, 0]

        try:
            cx1, cy1 = self.od.get_center_target(img, boundries[1])
            obj1 = self.od.get_object(img, boundries[1])
            prediction1 = self.svm.classify(obj1)
            data1 = [int(prediction1[1][0]), cx1, cy1] #prediction1[1][0],
        except:
            #self.targets.data = np.array([0, 0, 0, 0])
            data1 = [0, 0, 0]

        print("target 1 ", data0, " target 2 ", data1)


        # set variables for ublishing
        self.target = Float64MultiArray()
        self.target.data = np.array([data0, data1])
        # Publish the results
        try:
            #self.joints_pub.publish(self.joints)
            self.target_pub.publish(self.target)
        except CvBridgeError as e:
            print(e)

# call the class
def main(args):
    ic = image2target()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
