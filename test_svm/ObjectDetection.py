import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class ObjectDetection(object):

    #def __init__(self, img):
    #    self.img = img

    def filter_colour(self, img, colour):
        if colour == "orange":
            # filter orange colour
            mask = cv2.inRange(img, (75, 100, 120), (110, 190, 215))
        elif colour == "red":
            # filter red colour
            mask = cv2.inRange(img, (0, 0, 100), (0, 0, 255))
        elif colour == "green":
            # filter green colour
            mask = cv2.inRange(img, (0, 100, 0), (0, 255, 0))
        elif colour == "blue":
            # filter blue colour
            mask = cv2.inRange(img, (100, 0, 0), (255, 0, 0))
        elif colour == "yellow":
            # filter blue colour
            mask = cv2.inRange(img, (0, 100, 100), (0, 255, 255))
        return mask

    def dilate(self, img, kernel_size=3):
        # create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # dilate with kernel
        img = cv2.dilate(img, kernel, iterations=2)
        return img

    def opening(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size),np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations = 1)
        return img

    def find_boundries(self, img):
        # find contours
        img2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # boundries
        boundries = [cv2.boundingRect(ctr) for ctr in contours]
        return boundries, contours

    def get_center_joint(self, img):
        # compute moments
        M = cv2.moments(img)
        # calculate x,y coordinate of center
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    def get_center_target(self, img, boundries):
        # cut object out of image
        obj = img[boundries[1]-1:boundries[1]+boundries[3]+1, boundries[0]-1:boundries[0]+boundries[2]+1]
        # compute moments
        M = cv2.moments(obj)
        # calculate x,y coordinate of center
        cx = int(M["m10"] / M["m00"]) + boundries[0]
        cy = int(M["m01"] / M["m00"]) + boundries[1]
        return cx, cy

    def get_object(self, img, rect):
        '''
            rectangular region -> cut -> delete other objets from image -> resize
            in:     img           (grayscale image)
                    rects           (coordinates of detected objects)
            out:    obj             (object image)
        '''
        # Draw the rectangles
        #cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        #roi = img[pt1:pt1+leng, pt2:pt2+leng]

        roi_1 = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        roi_2 = img[pt1:pt1+leng, pt2:pt2+leng]

        # check if object is bigger then a certain size
        if (roi_2.shape[0] >= 3) & (roi_2.shape[1] >= 3):

            # make border
            BLACK = [0, 0, 0]

            # define borders
            y_ax = int((roi_2.shape[0] - roi_1.shape[0])/2) if roi_2.shape[0] > roi_1.shape[0] else 0
            x_ax = int((roi_2.shape[1] - roi_1.shape[1])/2) if roi_2.shape[1] > roi_1.shape[1] else 0

            # make borders around element
            obj = cv2.copyMakeBorder(roi_1, y_ax, y_ax, x_ax, x_ax, cv2.BORDER_CONSTANT, value=BLACK)
            #print(roi.shape)

            obj = cv2.resize(obj, (32, 32))#, interpolation=cv2.INTER_AREA)
            obj = cv2.dilate(obj, (3, 3))
        else:
            obj = np.zeros((32,32))

        return obj
