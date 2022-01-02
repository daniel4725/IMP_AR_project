import cv2
import threading
import numpy as np
from sklearn.mixture import GaussianMixture
import time
import pickle
import matplotlib.pyplot as plt


class Calibration:
    def __init__(self):
        self.flag = 0
        self.GMM_Image = np.zeros((400, 400))
        self.GMM_Model = None

    def timer_sec(self):
        time.sleep(1)
        self.flag += 1
        time.sleep(1)
        self.flag += 1
        time.sleep(1)
        self.flag += 1
        time.sleep(1)
        self.flag += 1

    def capture_hand(self):
        cap = cv2.VideoCapture(0)
        suc, prev = cap.read()
        prev = cv2.flip(prev, 1)
        cv2.rectangle(prev, (400, 140), (620, 360), (0, 255, 0), 0)
        cropPrev = prev[141:359, 401:619]
        prevGray = cv2.cvtColor(cropPrev, cv2.COLOR_BGR2GRAY)
        shape = cropPrev.shape
        numPixels = shape[0]*shape[1]
        tol = numPixels/2
        change = 0
        startTimer = 0
        t1 = threading.Thread(target=self.timer_sec)
        while cap.isOpened():
            suc, img = cap.read()
            img = cv2.flip(img, 1)
            cv2.rectangle(img, (400, 140), (620, 360), (0, 255, 0), 0)
            cropImg = img[141:359, 401:619]
            cv2.imshow('image', img)
            imgGray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            subImg = imgGray - prevGray
            y = subImg.reshape(1, -1)
            change = (y > 127).sum()
            if change >= tol:
                startTimer += 1
            if startTimer == 1:
                t1.start()
            if self.flag == 1:
                cv2.putText(img, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
            if self.flag == 2:
                cv2.putText(img, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
            if self.flag == 3:
                cv2.putText(img, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
            if self.flag == 4:
                cv2.putText(img, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.GMM_Image = cropImg
                cv2.imshow('image', img)
                cv2.waitKey(5)
                break
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def gmm_train(self):
        Shape = self.GMM_Image.shape
        imageLAB = cv2.cvtColor(self.GMM_Image, cv2.COLOR_BGR2LAB)

        L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()
        data = np.array([a, b]).transpose()

        n_compononts = 2
        self.GMM_Model = GaussianMixture(n_components=n_compononts)
        self.GMM_Model.fit(data)
        GMM_Labels = self.GMM_Model.predict(data)

        # save the model to disk
        filename = 'hand_gmm_model.sav'
        pickle.dump(self.GMM_Model, open(filename, 'wb'))

        # to show the results
        segmented = np.array(GMM_Labels).reshape(Shape[0], Shape[1])

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(segmented)
        ax.set_title(f"Segmented with {n_compononts} components")
        ax = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(cv2.cvtColor(self.GMM_Image, cv2.COLOR_BGR2RGB))
        ax.set_title('Original')
        plt.show()


if __name__ == "__main__":
    cal = Calibration()
    cal.capture_hand()
    cal.gmm_train()

