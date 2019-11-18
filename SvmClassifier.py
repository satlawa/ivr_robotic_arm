import cv2
import numpy as np
import matplotlib.pyplot as plt

# This code uses support vector machine on the flat part images
# There are three classes; part1, part2, and part3
# These training samples are stored in the folder train_images
# There is a test image in the directory test_image which is part2
#
# This code is under the assumption that the images are black and
# white

class classifer():

    # classes should be a 2D array where
    # the first value is the name of the class,
    # and the second is the number of training
    # instances
    def __init__(self, classes, no_dim=3):
        self.svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR,
                               svm_type=cv2.ml.SVM_C_SVC,
                               C=2.67, gamma=5.383)
        self.classes = classes
        self.dim = no_dim

        self.samples = np.array([[]], dtype=np.float32)
        self.samples_labels = np.array([], dtype=np.int)
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))

    def addTrainSamples(self, folder):
        class_key = 0
        for c in self.classes:
            for i in range(0, c[1]):
                filename = folder + '/' + c[0] + '_' + str(i) + '.jpg'
                img = cv2.imread(filename, 0)
                ret, img_bin = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                moment = cv2.moments(img)
                img_props = cv2.HuMoments(moment)
                img_props = img_props.reshape((1, 7)).astype(np.float32)
                if self.samples.size == 0:
                    self.samples = img_props
                else:
                    self.samples = np.append(self.samples, img_props, axis=0)
                self.samples_labels = np.append(self.samples_labels, np.array([class_key], dtype=np.int))
            class_key += 1
        return

    def train(self, model="", load=False):
        if load:

            try:
                self.svm = self.svm.load(model)
            except:
                print("Provide a valid xml file.")
        else:
            self.svm.train(self.samples, cv2.ml.ROW_SAMPLE, self.samples_labels)
            if model != "":
                try:
                    self.svm.save(model)
                except:
                    print("The filename must be valid.")
        return

    def classify(self, img_bin):
        #img = cv2.imread(filename, 0)
        #ret, img_bin = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        moment = cv2.moments(img_bin)
        img_props = cv2.HuMoments(moment)
        img_props = img_props.reshape((1, 7)).astype(np.float32)
        prediction = self.svm.predict(img_props)
        return prediction


def main():
    classes = [['obj0', 8], ['obj1', 8]]
    test_classifier = classifer(classes)

    ## APPROACH 1 ##
    # The following code can be used if there is not
    # a model to be loaded and must create one from
    # scratch.
    test_classifier.addTrainSamples('images_train')
    #test_classifier.train()

    # If you want to save the model to an xml file
    # then call the train method initialsing a model
    # name.
    test_classifier.train('model_name.xml')

    ## APPROACH 2 ##
    # The following code can be used if there is a
    # model that can be loaded. This means that you
    # do not need to add training samples. Replace
    # name_of_model with your desired model.
    #test_classifier.train('svm_01.xml', True)

    prediction = test_classifier.classify('images_test/test_obj1_0.jpg')
    print(prediction)
    prediction = test_classifier.classify('images_test/test_obj0_0.jpg')
    print(prediction)
    return


if __name__ == "__main__":
    main()
