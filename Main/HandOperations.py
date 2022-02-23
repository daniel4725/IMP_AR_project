import cv2
import numpy as np
import matplotlib.pyplot as plt
import ImageOperations
import pickle
import os


class HandOperations:
    def __init__(self):
        self.image = None
        self.mask = None
        self.numfingers = None
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_output")

        with open(os.path.join(directory, 'bovw_90_new_circle.pkl'), 'rb') as pickle_file:
            self.kmeans, self.scale, self.svm, self.im_features, self.train_labels, self.no_clusters = pickle.load(pickle_file)
        with open(os.path.join(directory, "hand_gmm_model.sav"), 'rb') as handle:
            self.GMM_Model = pickle.load(handle)
        with open(os.path.join(directory, "hand_best_labels.sav"), 'rb') as handle:
            self.hand_comp_label_list = pickle.load(handle)
        with open(os.path.join(directory, "hand_sleeve_labels.sav"), 'rb') as handle:
            self.sleeve_comp_label_list = pickle.load(handle)

        self.sift = cv2.SIFT_create()

        self.name_dict = {
            "0": "menu",
            "1": "triangle",
            "2": "circle",
            "3": "square",
        }

    def get_two_comp_segmentation(self, n_comp_segmented_img: np.array, get_sleeve=False):
        reduced_GMM_Labels_segmented_img = np.zeros(n_comp_segmented_img.shape)
        if get_sleeve:
            reduced_GMM_Labels_segmented_img[np.isin(n_comp_segmented_img, self.sleeve_comp_label_list)] = 1
        else:
            reduced_GMM_Labels_segmented_img[np.isin(n_comp_segmented_img, self.hand_comp_label_list)] = 1
        return reduced_GMM_Labels_segmented_img

    def get_segmentation(self, img: np.array, get_sleeve=False):
        Shape = img.shape
        img = cv2.resize(img, (Shape[1]//2, Shape[0]//2), interpolation=cv2.INTER_AREA)
        imageLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()
        data = np.array([a, b]).transpose()

        GMM_Labels = self.GMM_Model.predict(data)

        segmented_labels = np.array(GMM_Labels).reshape(img.shape[0], img.shape[1])

        segmented = self.get_two_comp_segmentation(segmented_labels, get_sleeve)

        segmented = cv2.resize(segmented, (Shape[1], Shape[0]), interpolation=cv2.INTER_AREA)
        return segmented

    def get_hand_mask(self, image: np.array, get_sleeve=False):
        predicted = self.get_segmentation(image, get_sleeve)
        imageseg = ImageOperations.morph_close(predicted)
        imageseg = ImageOperations.morph_open(imageseg)
        imagenorm = cv2.normalize(imageseg, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        blurred = cv2.GaussianBlur(imagenorm, (3, 3), cv2.BORDER_DEFAULT)
        contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(image.shape, dtype='uint8')
        if len(contours) > 0:
            maxcontour = max(contours, key=lambda x: cv2.contourArea(x))
            cv2.drawContours(mask, maxcontour, -1, (255, 255, 255), 1)
            cv2.fillPoly(mask, pts=[maxcontour], color=(255, 255, 255))
        self.mask = mask
        return mask[:, :, 0]

    def finger_count(self, imageseq: np.array):
        # imageseg = self.get_hand_mask(image=image)

        # might be error because imageseq has one channel so findcontour might crush
        contours, hierarchy = cv2.findContours(imageseq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            maxcontour = max(contours, key=lambda x: cv2.contourArea(x))
            extTop = tuple(maxcontour[maxcontour[:, :, 1].argmin()][0])
            extBot = tuple(maxcontour[maxcontour[:, :, 1].argmax()][0])
            maxdis = np.sqrt((extTop[0] - extBot[0]) ** 2 + (extTop[1] - extBot[1]) ** 2)
            hull = cv2.convexHull(maxcontour)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(maxcontour)
            arearatio = ((areahull - areacnt) / areacnt) * 100
            hull = cv2.convexHull(maxcontour, returnPoints=False)
            defects = cv2.convexityDefects(maxcontour, hull)
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(maxcontour[s][0])
                    end = tuple(maxcontour[e][0])
                    far = tuple(maxcontour[f][0])
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    leght_ratio = max(a, b)/maxdis
                    if angle <= np.pi / 2 and leght_ratio >= 0.25:  # angle less than 90 degree, treat as fingers
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

    def predict_shape(self, image: np.array):
        imageResized = cv2.resize(image, (150, 150))
        kp, des = self.sift.detectAndCompute(imageResized, None)
        pre_features = [np.bincount(self.kmeans.predict(des), minlength=self.no_clusters)]
        pre_features = self.scale.transform(pre_features)
        prediction = self.svm.predict(pre_features)
        prediction = int(prediction[0])
        # prediction = self.name_dict[str(prediction)]

        """
            "0": "smilie",
            "1": "triangle",
            "2": "circle",
            "3": "square",
        """
        predict_dict = {0: 4, 1: 1, 2: 2, 3: 3}
        return predict_dict[prediction]

    # def capture_image(self):
    #     cap = cv2.VideoCapture(0)
    #     suc, prev = cap.read()
    #     prev = cv2.flip(prev, 1)
    #     roi = [140, 360, 400, 620]  # [y_start, y_end, x_start, x_end]
    #     cv2.rectangle(prev, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)
    #     cropPrev = prev[roi[0] + 1:roi[1] - 1, roi[2] + 1:roi[3] - 1]
    #     shape = cropPrev.shape
    #     prevMask = self.get_hand_mask(cropPrev)
    #     # prevGray = cv2.cvtColor(cropPrev, cv2.COLOR_BGR2GRAY)
    #     numPixels = shape[0] * shape[1]
    #     tol = numPixels / 50
    #     change = 0
    #     startTimer = 0
    #     cal = Calibration()
    #     t1 = threading.Thread(target=cal.timer_sec)
    #     while cap.isOpened():
    #         suc, img = cap.read()
    #         # img = img[:, 0:int(img.shape[1] / 2), :]  # left image
    #         img = cv2.flip(img, 1)
    #         cv2.rectangle(img, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)
    #         cropImg = img[roi[0]+1:roi[1]-1, roi[2]+1:roi[3]-1]
    #         imgMask = self.get_hand_mask(cropImg)
    #         # imgGray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
    #         subImg = cv2.subtract(imgMask, prevMask)
    #         y = subImg.reshape(1, -1)
    #         change = (y >= 1).sum()
    #         if change >= tol:
    #             startTimer += 1
    #         if startTimer == 1:
    #             t1.start()
    #         if cal.flag == 1:
    #             cv2.putText(img, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             cv2.imshow('image', img)
    #         if cal.flag == 2:
    #             cv2.putText(img, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             cv2.imshow('image', img)
    #         if cal.flag == 3:
    #             cv2.putText(img, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             cv2.imshow('image', img)
    #         if cal.flag == 4:
    #             cv2.putText(img, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             capturedImg = cropImg
    #             self.image = img
    #             cv2.imshow('image', img)
    #             cv2.waitKey(5)
    #             break
    #         cv2.imshow('image', img)
    #         cv2.imshow('prev', prevMask)
    #         cv2.imshow('seg', imgMask)
    #         cv2.imshow('sub', subImg)
    #         key = cv2.waitKey(5)
    #         if key == ord('q'):
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     return capturedImg

    def get_finger_num(self, imageseq):
        self.finger_count(imageseq=imageseq)
        if self.numfingers > 5:
            self.numfingers = 0
        return self.numfingers

    def get_tip_mask(self, mask):
        thick = 15
        ys, xs = np.where(mask)
        points = np.array([xs, ys]).T
        center_x = xs.mean()
        center_y = ys.mean()
        center = np.array([[center_x, center_y]])
        distances = ((points - center) ** 2).sum(axis=1) ** 0.5
        if distances.shape[0] == 0:
            return np.zeros_like(mask)
        tip = points[np.argmax(distances)]
        tip_map = np.zeros_like(mask)

        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0]
        # hull = cv2.convexHull(contours, returnPoints=False)
        # defects = cv2.convexityDefects(contours, hull)
        # max_defect = max
        # max_def = defects[np.argmax(defects[:, :, 3]), 0, :]
        # s, e, f, d = max_def
        # # start = contours[s][0]
        # end = contours[e][0]
        # far = contours[f][0]
        # distances_end = ((end - center) ** 2).sum() ** 0.5
        # distances_far = ((far - center) ** 2).sum() ** 0.5
        # if distances_end > distances_far:
        #     point = end
        # else:
        #     point = far
        return cv2.circle(tip_map, tuple(tip), thick, 1, -1)

if __name__ == "__main__":
    hand = HandOperations()
    # test = cv2.imread('test_shape.png')
    # hand.get_shape(img_to_predict=test, single=True)
    # cap = cv2.VideoCapture('seqmented.mp4')
    # suc, frame = cap.read()
    # shape = frame.shape
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('output.mp4', fourcc, fps, (shape[1], shape[0]))
    # total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # skip_step = 1
    # i = 0
    # while cap.isOpened():
    #     suc, frame = cap.read()
    #     i += 1
    #     # if (i % skip_step) == i:
    #     if suc:
    #         perdiction = hand.get_shape(img_to_predict=frame, single=True)
    #         cv2.putText(frame, perdiction, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #         cv2.imshow('Stream', frame)
    #         out.write(frame)
    #         if cv2.waitKey(50) & 0xFF == ord('q'):
    #             break
    #     else:
    #             break
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(1)
    # while cap.isOpened():
    #     suc, frame = cap.read()
    #     seq = hand.get_hand_mask(image=frame)
    #     show = hand.finger_count(imageseq=seq)
    #     print(hand.numfingers)
    #     cv2.imshow('seq', show)
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(50) & 0xFF == ord('q'):
    #         cap.release()
    #         cv2.destroyAllWindows()





