import cv2
import numpy as np

zero_image = np.zeros((360,640)).astype(np.uint8)
final_image = np.zeros((360,640)).astype(np.uint8)

def hand_contour(image: np.array):
    zero_image = np.zeros(image.shape).astype(np.uint8)
    roi = [zero_image.shape[0] - 340, zero_image.shape[0] - 120, zero_image.shape[1] - 240, zero_image.shape[1] - 20]  # [y_start, y_end, x_start, x_end]
    cropPrev = zero_image[roi[0]:roi[1], roi[2]:roi[3]]
    crop_shape = cropPrev.shape
    handExample = cv2.imread('handExample.jpeg')
    handExample = cv2.resize(handExample, (crop_shape[1], crop_shape[0]))
    handExampleGray = cv2.cvtColor(handExample, cv2.COLOR_BGR2GRAY)
    zero_image[roi[0]:roi[1], roi[2]:roi[3]] = 255 - handExampleGray
    ret, mask = cv2.threshold(zero_image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours[0]

def draw_contour(image: np.array, contour: np.array):
    cv2.drawContours(image, contour, -1, (0,0,255), 3)
    
contour = hand_contour(final_image)
draw_contour(final_image, contour)
    
cv2.imshow("test", final_image); cv2.waitKey(0); cv2.destroyAllWindows();

print("test")