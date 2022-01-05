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
        handExample = cv2.imread('handExample.jpeg')
        cap = cv2.VideoCapture(1)
        while True:
            suc, prev = cap.read()
            if suc == True:
                break
        prev = prev[:, 0:int(prev.shape[1]/2), :]  # left image
        # prev = cv2.flip(prev, 1)
        roi = [140, 360, 400, 620]  # [y_start, y_end, x_start, x_end]
        cv2.rectangle(prev, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)
        cropPrev = prev[roi[0]+1:roi[1]-1, roi[2]+1:roi[3]-1]
        shape = cropPrev.shape
        handExample = cv2.resize(handExample, (int(shape[1]/2), shape[0]))
        handExampleGray = cv2.cvtColor(handExample, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(handExampleGray, 10, 255, cv2.THRESH_BINARY)
        mask = np.array(np.hstack([mask, mask]))
        prevGray = cv2.cvtColor(cropPrev, cv2.COLOR_BGR2GRAY)
        numPixels = shape[0]*shape[1]
        tol = numPixels/2
        change = 0
        startTimer = 0
        t1 = threading.Thread(target=self.timer_sec)
        while cap.isOpened():
            suc, img = cap.read()
            img = img[:, 0:int(img.shape[1] / 2), :]  # left image
            # img = cv2.flip(img, 1)
            imgView = np.array(img, copy=True)
            cv2.rectangle(imgView, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)
            cropImg = img[roi[0]+1:roi[1]-1, roi[2]+1:roi[3]-1]
            cropImgView = imgView[roi[0]+1:roi[1]-1, roi[2]+1:roi[3]-1]
            cropImgView = cv2.bitwise_and(cropImgView, cropImgView, mask=mask)
            imgView[roi[0]+1:roi[1]-1, roi[2]+1:roi[3]-1] = cropImgView
            imgGray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            subImg = imgGray - prevGray
            y = subImg.reshape(1, -1)
            change = (y > 127).sum()
            if change >= tol:
                startTimer += 1
            if startTimer == 1:
                t1.start()
            if self.flag == 1:
                cv2.putText(imgView, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', imgView)
            if self.flag == 2:
                cv2.putText(imgView, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', imgView)
            if self.flag == 3:
                cv2.putText(imgView, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', imgView)
            if self.flag == 4:
                cv2.putText(imgView, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.GMM_Image = cropImg
                cv2.imshow('image', imgView)
                cv2.waitKey(5)
                break
            cv2.imshow('image', imgView)
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

        if GMM_Labels[0] == 1:
            GMM_Labels = np.array(GMM_Labels, dtype=bool)
            GMM_Labels = np.invert(GMM_Labels)
            GMM_Labels = np.array(GMM_Labels, dtype=int)

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
    # all_camera_idx_available = []
    #
    # for camera_idx in range(10):
    #     cap = cv2.VideoCapture(camera_idx)
    #     if cap.isOpened():
    #         print(f'Camera index available: {camera_idx}')
    #         all_camera_idx_available.append(camera_idx)
    #         cap.release()


    cal = Calibration()
    cal.capture_hand()
    cal.gmm_train()

