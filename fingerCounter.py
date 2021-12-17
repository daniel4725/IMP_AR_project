import cv2
import numpy as np


cap = cv2.VideoCapture(0)  # '0' for webcam
while cap.isOpened():
    _, img = cap.read()
    try:
        cv2.rectangle(img, (50, 170), (220, 340), (0, 255, 0), 0)
        crop_img = img[171:339, 51:219]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        blurred = cv2.GaussianBlur(grey, (5, 5), cv2.BORDER_DEFAULT)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # threshold = 80
        # canny_output = cv2.Canny(grey, threshold, threshold * 2)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsMax = max(contours, key=lambda x: cv2.contourArea(x))
        # largeContour = np.vstack([cntsMax[-1], cntsMax[-2]])
        cv2.drawContours(crop_img, cntsMax, -1, (255, 255, 0), 2)
        hull1 = cv2.convexHull(cntsMax)
        cv2.drawContours(crop_img, [hull1], -1, (0, 255, 255), 2)
        areahull = cv2.contourArea(hull1)
        areacnt = cv2.contourArea(cntsMax)
        arearatio = ((areahull - areacnt) / areacnt) * 100
        hull = cv2.convexHull(cntsMax, returnPoints=False)
        defects = cv2.convexityDefects(cntsMax, hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(cntsMax[s][0])
                end = tuple(cntsMax[e][0])
                far = tuple(cntsMax[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(crop_img, far, 4, [0, 0, 255], -1)
            if cnt > 0:
                cnt = cnt + 1
            if cnt == 0 and arearatio >= 12:
                cnt = cnt + 1
            cv2.putText(img, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', img)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()