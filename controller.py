import sys
import cv2
from image1 import image_converter1
from image2 import image_converter2
from ObjectDetection import ObjectDetection
from SvmClassifier import classifer


class controller:

    def __init__(self):
        self.ic1 = image_converter1()
        self.ic2 = image_converter2()
        self.od = ObjectDetection()
        self.svm = classifer([['obj0', 8], ['obj1', 8]])
        self.svm.addTrainSamples('images_train')
        self.svm.train('svm.xml')

    def detect_joints(self, img):
        mask = self.ic.filter_color(img, "blue")
        mask = self.od.dilate(img=mask, kernel_size=5)
        return mask

    def detect_target(self, img):
        img = self.od.filter_colour(img, "orange")
        img = self.od.opening(img, kernel_size=3)
        img = self.od.dilate(img, 3)
        boundries, contours = self.od.find_boundries(img)
        return img, boundries, contours

def main(args):
    con = controller()
    i1 = con.ic1.iterator
    i2 = con.ic2.iterator

    while True:
        if con.ic1.iterator > i1:
            print("camera 1")
            print(con.ic1.iterator)
            i1 = con.ic1.iterator

            # get image1
            img = con.ic1.cv_image1
            # detect targets
            img, boundries, contours = con.detect_target(img)
            try:
                # get center
                cx0, cy0 = con.od.get_center(img, boundries[0])
                obj0 = con.od.get_object(img, boundries[0])
                prediction0 = con.svm.classify(obj0)
                print(prediction0[1][0], cx0, cy0)
                #print(prediction, cx, cy)
            except:
                #self.targets.data = np.array([0, 0, 0, 0])
                print("x x x")

            try:
                cx1, cy1 = con.od.get_center(img, boundries[1])
                obj1 = con.od.get_object(img, boundries[1])
                prediction1 = con.svm.classify(obj1)
                print(prediction1[1][0], cx1, cy1)
            except:
                #self.targets.data = np.array([0, 0, 0, 0])
                print("x x x")

        if con.ic2.iterator > i2:
            print("camera 2")
            print(con.ic2.iterator)
            i2 = con.ic2.iterator

            # get image1
            img = con.ic2.cv_image2
            # detect targets
            img, boundries, contours = con.detect_target(img)
            try:
                # get center
                cx0, cy0 = con.od.get_center(img, boundries[0])
                obj0 = con.od.get_object(img, boundries[0])
                prediction0 = con.svm.classify(obj0)
                print(prediction0[1][0], cx0, cy0)
                #print(prediction, cx, cy)
            except:
                #self.targets.data = np.array([0, 0, 0, 0])
                print("x x x")

            try:
                cx1, cy1 = con.od.get_center(img, boundries[1])
                obj1 = con.od.get_object(img, boundries[1])
                prediction1 = con.svm.classify(obj1)
                print(prediction1[1][0], cx1, cy1)
            except:
                #self.targets.data = np.array([0, 0, 0, 0])
                print("x x x")



# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
