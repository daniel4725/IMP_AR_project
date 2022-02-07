import cv2
import numpy as np
import matplotlib.pyplot as plt
from Main.Calibration import Calibration
import ImageOperations
import pickle
import threading


class HandOperations:
    def __init__(self):
        self.image = None
        self.mask = None
        self.maxcontour = None
        self.numfingers = None
        self.GMM_Model = None
        self.two_comp_label_list = []
        self.contours = None

        with open('bovw.pkl', 'rb') as pickle_file:
            self.kmeans, self.scale, self.svm, self.im_features, self.train_labels, self.no_clusters = pickle.load(pickle_file)

        self.name_dict = {
            "0": "menu",
            "1": "triangle",
            "2": "circle",
            "3": "square",
        }

    def load_saved_model(self, path: str):
        with open(path, 'rb') as handle:
            self.GMM_Model = pickle.load(handle)

    def load_saved_best_labels(self, path: str):
        with open(path, 'rb') as handle:
            self.two_comp_label_list = pickle.load(handle)

    def get_two_comp_segmentation(self, n_comp_segmented_img: np.array):
        reduced_GMM_Labels_segmented_img = np.zeros(n_comp_segmented_img.shape)
        reduced_GMM_Labels_segmented_img[np.isin(n_comp_segmented_img, self.two_comp_label_list)] = 1
        return reduced_GMM_Labels_segmented_img

    def get_segmentation(self, img: np.array):
        Shape = img.shape
        imageLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()
        data = np.array([a, b]).transpose()

        GMM_Labels = self.GMM_Model.predict(data)

        segmented_labels = np.array(GMM_Labels).reshape(Shape[0], Shape[1])

        segmented = self.get_two_comp_segmentation(segmented_labels)
        return segmented

    def get_hand_mask(self, image: np.array):
        predicted = self.get_segmentation(image)
        imagedim = image.shape
        imageseg = np.array(predicted).reshape(imagedim[0], imagedim[1])
        imageseg = ImageOperations.morph_open(imageseg)
        imagenorm = cv2.normalize(imageseg, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        blurred = cv2.GaussianBlur(imagenorm, (5, 5), cv2.BORDER_DEFAULT)
        self.contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(imagedim)
        if len(self.contours) > 0:
            self.maxcontour = max(self.contours, key=lambda x: cv2.contourArea(x))
            cv2.drawContours(mask, self.maxcontour, -1, (255, 255, 255), 1)
            cv2.fillPoly(mask, pts=[self.maxcontour], color=(255, 255, 255))
        self.mask = mask
        return mask[:, :, 0]

    def finger_count(self, image: np.array):
        imageseg = self.get_hand_mask(image=image)
        if len(self.contours) > 0:
            hull = cv2.convexHull(self.maxcontour)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(self.maxcontour)
            arearatio = ((areahull - areacnt) / areacnt) * 100
            hull = cv2.convexHull(self.maxcontour, returnPoints=False)
            defects = cv2.convexityDefects(self.maxcontour, hull)
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(self.maxcontour[s][0])
                    end = tuple(self.maxcontour[e][0])
                    far = tuple(self.maxcontour[f][0])
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                if cnt > 0:
                    cnt = cnt + 1
                if cnt == 0 and arearatio >= 12:
                    cnt = cnt + 1
                self.numfingers = cnt
        else:
            self.numfingers = 0

    def preview(self, funname: str):
        preview = np.array(self.mask, copy=True)
        if funname == 'finger_count':
            cv2.putText(preview, str(self.numfingers), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            cv2.imshow('flow', preview)
            key = cv2.waitKey(5)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

        if funname == 'get_hand_mask':
            cv2.imshow('flow', preview)
            key = cv2.waitKey(5)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

        if funname == 'preview':
            preview = self.image
            cv2.imshow('flow', preview)
            key = cv2.waitKey(5)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

    def center_of_mass(self):
        M = cv2.moments(self.maxcontour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    def predict_shape(self, image: np.array):
        imageResized = cv2.resize(image, (500, 500))
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(imageResized, None)
        pre_features = np.zeros(self.no_clusters)
        for i in range(len(des)):
            feature = des[i]
            feature = feature.reshape(1, 128)
            idx = self.kmeans.predict(feature)
            pre_features[idx] += 1
        pre_features = [pre_features]
        pre_features = self.scale.transform(pre_features)
        prediction = self.svm.predict(pre_features)
        prediction = int(prediction[0])
        prediction = self.name_dict[str(prediction)]
        return prediction

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        suc, prev = cap.read()
        prev = cv2.flip(prev, 1)
        roi = [140, 360, 400, 620]  # [y_start, y_end, x_start, x_end]
        cv2.rectangle(prev, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)
        cropPrev = prev[roi[0] + 1:roi[1] - 1, roi[2] + 1:roi[3] - 1]
        shape = cropPrev.shape
        prevMask = self.get_hand_mask(cropPrev)
        # prevGray = cv2.cvtColor(cropPrev, cv2.COLOR_BGR2GRAY)
        numPixels = shape[0] * shape[1]
        tol = numPixels / 50
        change = 0
        startTimer = 0
        cal = Calibration()
        t1 = threading.Thread(target=cal.timer_sec)
        while cap.isOpened():
            suc, img = cap.read()
            # img = img[:, 0:int(img.shape[1] / 2), :]  # left image
            img = cv2.flip(img, 1)
            cv2.rectangle(img, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)
            cropImg = img[roi[0]+1:roi[1]-1, roi[2]+1:roi[3]-1]
            imgMask = self.get_hand_mask(cropImg)
            # imgGray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            subImg = cv2.subtract(imgMask, prevMask)
            y = subImg.reshape(1, -1)
            change = (y >= 1).sum()
            if change >= tol:
                startTimer += 1
            if startTimer == 1:
                t1.start()
            if cal.flag == 1:
                cv2.putText(img, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
            if cal.flag == 2:
                cv2.putText(img, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
            if cal.flag == 3:
                cv2.putText(img, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', img)
            if cal.flag == 4:
                cv2.putText(img, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                capturedImg = cropImg
                self.image = img
                cv2.imshow('image', img)
                cv2.waitKey(5)
                break
            cv2.imshow('image', img)
            cv2.imshow('prev', prevMask)
            cv2.imshow('seg', imgMask)
            cv2.imshow('sub', subImg)
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return capturedImg

    def get_shape(self, img_to_predict: np.array, single: bool = False):
        if single:
            perdiction = self.predict_shape(image=img_to_predict)
        else:
            img = self.capture_image()
            perdiction = self.predict_shape(image=img)
        # cv2.putText(self.image, perdiction, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow('image', self.image)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     cv2.destroyAllWindows()
        print(perdiction)
        return perdiction

    def get_finger_num(self):
        img = self.capture_image()
        self.finger_count(image=img)
        # cv2.putText(self.image, str(self.numfingers), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow('image', self.image)
        # key = cv2.waitKey(0)
        # if key == ord('q'):
        #     cv2.destroyAllWindows()
        print(self.numfingers)


if __name__ == "__main__":
    hand = HandOperations()
    test = cv2.imread('test_shape.png')
    hand.get_shape(img_to_predict=test, single=True)
    cap = cv2.VideoCapture('regular_video_right.mp4')
    suc, frame = cap.read()
    shape = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (shape[1], shape[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_step = 2
    i = 0
    while cap.isOpened():
        suc, frame = cap.read()
        i += 1
        if (i % skip_step) == 0:
            if suc:
                perdiction = hand.get_shape(img_to_predict=frame, single=True)
                cv2.putText(frame, perdiction, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Stream', frame)
                out.write(frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            else:
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()





