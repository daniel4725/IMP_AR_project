import cv2
import numpy as np
import matplotlib.pyplot as plt
from Calibration import Calibration
import ImageOperations


class HandOperations:
    def __init__(self, image):
        self.image = image
        self.mask = None
        self.maxcontour = None
        self.numfingers = None

    def get_hand_mask(self, gmm_model):
        imagedim = self.image.shape

        imageLAB = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)

        L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()

        data = np.array([a, b]).transpose()

        predicted = gmm_model.predict(data)
        imageseg = np.array(predicted).reshape(imagedim[0], imagedim[1])
        imageseg = ImageOperations.morph_open(imageseg)
        imagenorm = cv2.normalize(imageseg, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        blurred = cv2.GaussianBlur(imagenorm, (5, 5), cv2.BORDER_DEFAULT)
        contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.maxcontour = max(contours, key=lambda x: cv2.contourArea(x))
        mask = np.zeros(imagedim)
        cv2.drawContours(mask, self.maxcontour, -1, (255, 255, 255), 1)
        cv2.fillPoly(mask, pts=[self.maxcontour], color=(255, 255, 255))
        self.mask = mask[:, :, 0]
        
        return np.array(self.mask, copy=True)

    def finger_count(self):
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


if __name__ == "__main__":
    cal = Calibration()
    cal.capture_hand()
    cal.gmm_train()
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        suc, img = cap.read()
        img = img[:, 0:int(img.shape[1] / 2), :]  # left image
        hand = HandOperations(image=img)
        # hand.get_hand_mask(gmm_model=cal.GMM_Model)
        # hand.finger_count()
        hand.preview('preview')
