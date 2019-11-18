
from SvmClassifier import classifer
import cv2
import numpy as np
import os

input_path = os.path.join(os.getcwd(), 'images_training')
path_model = os.path.join(os.getcwd(), 'svm.xml')
path_test = os.path.join(os.getcwd(), 'svm_01.xml')

classes = [['obj0', 8], ['obj1', 8]]

svm = classifer(classes)

svm.addTrainSamples('images_train')
svm.train('svm.xml')
#svm.train('svm.xml', True)

img = cv2.imread('images_test/test_obj0_0.jpg', 0)
ret, img_bin = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
print(img_bin.shape)
prediction = svm.classify(img_bin)
print(prediction)
