import cv2
import numpy as np


def show_image(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    c = cv2.waitKey()
    if c >= 0 : return -1
    return 0


image = cv2.imread('squre.jpg')
cv2.rectangle(image, (230, 320), (950, 1200), (0, 255, 0), 3)
crop_img = image[320:1200, 230:950]
img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
threshold = 60
canny_output = cv2.Canny(img_gray, threshold, threshold * 2)
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
largeContour = np.vstack([cntsSorted[-1], cntsSorted[-2], cntsSorted[-3]])
# res_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
cv2.drawContours(crop_img, contours, -1, (255,255,0), 2)
hull = cv2.convexHull(largeContour)
cv2.drawContours(crop_img, [hull], -1, (0, 255, 255), 2)
show_image(image)
