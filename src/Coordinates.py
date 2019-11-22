import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Coordinates(object):

    #def __init__(self, img):
    #    self.img = img

    def calc_pixel_to_meter_2D(self, joint0, joint1, dist_meter):
        dist_pix = np.sum((joint0 - joint1)**2)
        pix2meter = dist_meter/np.sqrt(dist_pix)
        return pix2meter

    def merge_coordinates_2D_to_3D(self, joint_yz, joint_xz):
        x = joint_xz[0]
        y = joint_yz[0]
        if joint_yz[1] == 0:
            z = -joint_xz[1]
        if joint_xz[1] == 0:
            z = -joint_yz[1]
        else:
            z = -(joint_xz[1] + joint_yz[1]) / 2
        return [x,y,z]

    def calc_angles_joints_3D(self, joints):
        ja1 = np.arctan2(joints[0][0]- joints[0][0], joints[0][1] - joints[0][1])
        ja2 = np.arctan2(joints[0][0]-joints[0][0], joints[0][1]-joints[0][1]) - ja1
        ja3 = np.arctan2(joints[0][0]-joints[0][0], joints[0][1]-joints[0][1]) - ja2 - ja1
        return np.array([ja1, ja2, ja3])
