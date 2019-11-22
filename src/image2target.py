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
from Coordinates import Coordinates


class image2target:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joints_pub = rospy.Publisher("joints_pos",Float64MultiArray, queue_size=10)
        self.target_x_pub = rospy.Publisher("/target/x_position_estimation", Float64, queue_size=10)
        self.target_y_pub = rospy.Publisher("/target/y_position_estimation", Float64, queue_size=10)
        self.target_z_pub = rospy.Publisher("/target/z_position_estimation", Float64, queue_size=10)
        #self.target_pub = rospy.Publisher("target_pos",Float64MultiArray, queue_size=10)
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
        self.coord = Coordinates()
        self.od = ObjectDetection()
        self.svm = classifer([['obj0', 8], ['obj1', 8]])
        self.svm.addTrainSamples('images_train')
        self.svm.train('svm.xml')

        self.pix2meter0 = 0
        self.pix2meter1 = 0

        self.base_pix_coord0 = np.array([0,0])
        self.base_pix_coord1 = np.array([0,0])

        self.target0 = {"circle": [0, 0], "rectangle": [0, 0]}
        self.target1 = {"circle": [0, 0], "rectangle": [0, 0]}

    def detect_joints(self, img):
        last_colour = "yellow"
        joints = []
        for colour in ["yellow", "blue", "green", "red"]:
            mask = self.od.filter_colour(img, colour)
            mask = self.od.dilate(img=mask, kernel_size=5)
            cx, cy = self.od.get_center_joint(mask)
            if cx == None:
                mask = self.od.filter_colour(img, last_colour)
                mask = self.od.dilate(img=mask, kernel_size=5)
                cx, cy = self.od.get_center_joint(mask)
            else:
                last_colour = colour
            joints.append([cx, cy])
        return np.array(joints)

    def detect_target(self, img):
        img = self.od.filter_colour(img, "orange")
        img = self.od.opening(img, kernel_size=3)
        img = self.od.dilate(img, 3)
        boundries, contours = self.od.find_boundries(img)
        return img, boundries, contours

    def classify_target(self, img, boundries):
        # classify target object | 0 = rectangle | 1 = circle
        obj = self.od.get_object(img, boundries)
        prediction = self.svm.classify(obj)
        return int(prediction[1][0])

    def set_pix2meter(self, joints0, joints1):
        # set conversion factor (pixels to meters) between green and blue joints (3 meters)
        self.pix2meter0 = self.coord.calc_pixel_to_meter_2D(joints0[1], joints0[2], 3)
        self.pix2meter1 = self.coord.calc_pixel_to_meter_2D(joints1[1], joints1[2], 3)

    def set_base_coordinates(self, joints0, joints1):
        # set base pixel coordinates
        self.base_pix_coord0 = joints0[0]
        self.base_pix_coord1 = joints1[0]

    def convert_coordinates(self, joints0, joints1):
        # convert image-pixel coordinates to base-merter coordinates
        joint_base0 = []
        for joint in joints0:
            joint_base0.append((joint - self.base_pix_coord0) * self.pix2meter0)
        joint_base1 = []
        for joint in joints1:
            joint_base1.append((joint - self.base_pix_coord1) * self.pix2meter1)

        return np.array(joint_base0), np.array(joint_base1)

    def merge_coordinates(self, joints0, joints1):
        coordinates = []
        for i in range(joints0.shape[0]):
            coordinates.append(self.coord.merge_coordinates_2D_to_3D(joints0[i], joints1[i]))
        return np.array(coordinates)


    # Recieve data from camera 1, process it, and publish
    def callback_sync(self, data0, data1):
        self.iterator += 1
        # Recieve the image
        try:
            self.cv_image0 = self.bridge.imgmsg_to_cv2(data0, "bgr8")
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
        except CvBridgeError as e:
            print(e)

        ######################################################################
        ###                         joints                                 ###
        ######################################################################

        # detect joints from camera1 ans camera2
        joints0 = self.detect_joints(self.cv_image0)
        joints1 = self.detect_joints(self.cv_image1)


        if self.pix2meter0 == 0:
            # set conversion factor between green and blue joints
            self.set_pix2meter(joints0, joints1)
            # set base pixel coordinates
            self.set_base_coordinates(joints0, joints1)

        # convert image-pixel coordinates to base-merter coordinates
        joints0, joints1 = self.convert_coordinates(joints0, joints1)

        # merge pixel coordinates from both images to coordinates x,y,z
        coordinates = self.merge_coordinates(joints0, joints1)

        # calculate angles
        #print(self.coord.calc_angles_joints_3D(coordinates))

        #print(coordinates[0], coordinates[1], coordinates[2], coordinates[3])

        ######################################################################
        ###                         targets                                ###
        ######################################################################

        # get filterd & thresholded image, boundries, contours of target objects
        img0, boundries0, contours0 = self.detect_target(self.cv_image0)
        img1, boundries1, contours1 = self.detect_target(self.cv_image1)

        for boundries in boundries0:
            prediction = self.classify_target(img0, boundries)
            cx0, cy0 = self.od.get_center_target(img0, boundries)
            if prediction == 1:
                self.target0["circle"] = [cx0, cy0]
            else:
                self.target0["rectangle"] = [cx0, cy0]


        #if len(boundries1) == 2:
        for boundries in boundries1:
            prediction = self.classify_target(img1, boundries)
            cx0, cy0 = self.od.get_center_target(img1, boundries)
            if prediction == 1:
                self.target1["circle"] = [cx0, cy0]
            else:
                self.target1["rectangle"] = [cx0, cy0]


        #circle = np.array([self.target0["circle"], self.target1["circle"]])
        cir_coord0, cir_coord1 = self.convert_coordinates(np.array([self.target0["circle"]]), np.array([self.target1["circle"]]))
        cir_coordinates = self.merge_coordinates(cir_coord0, cir_coord1)

        print(cir_coordinates)

        # Publish the results
        try:
            #self.joints_pub.publish(self.joints)
            self.target_x_pub.publish(cir_coordinates[0][0])
            self.target_y_pub.publish(cir_coordinates[0][1])
            self.target_z_pub.publish(cir_coordinates[0][2])
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
