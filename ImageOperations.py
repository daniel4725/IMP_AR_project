import cv2
import numpy as np


def morph_open(image):
    image = cv2.normalize(image, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    kernel = cv2.getStructuringElement(cv2.MARKER_CROSS, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def morph_close(image):
    image = cv2.normalize(image, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image
