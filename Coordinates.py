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

    def merge_coordinates_2D_to_3D(self, joint_xz, joint_yz):
        x = joint_xz[0]
        y = joint_yz[0]
        z = (joint_xz[1] + joint_yz[1]) / 2
        return [x,y,z]

    def calc_angles_joints_3D(self, joint0, joint1):
        ang_xz = np.arctan2(joint1[2] - joint0[2], joint1[0] - joint0[0] + np.pi / 2) % np.pi
        ang_yz = np.arctan2(joint1[2] - joint0[2], joint1[1] - joint0[1] + np.pi / 2) % np.pi
        return np.array(ang_xz, ang_yz)
