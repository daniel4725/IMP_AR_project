import numpy as np
import cv2
import time
from matplotlib import pyplot as plt



def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr




def draw_contour_masked(img,flow, step=16):
    try:
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]

        v = np.sqrt(fx*fx+fy*fy)

        contour = np.zeros((h, w, 1), np.uint8)

        contour[v > 3] = 255
        
        blurred = cv2.GaussianBlur(contour, (5, 5), cv2.BORDER_DEFAULT)
        
        # threshold = 80
        # canny_output = cv2.Canny(blurred, threshold, threshold * 2)

        # get the largest contour
        contours = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        big_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(big_contour)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
        
        epsilon = 0.1*cv2.arcLength(big_contour,True)
        approx = cv2.approxPolyDP(big_contour,epsilon,True)

        # draw white contour on black background as mask
        mask = np.zeros((h, w), dtype=np.uint8)
        # cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), cv2.FILLED)
        
        hull = cv2.convexHull(big_contour)
        area = cv2.contourArea(hull)
        if (area > convex_size_q.calc_mean()):
            if (area - convex_size_q.last_area()) < 25082.0:
                convex_size_q.insert((area,hull))
        elif (convex_size_q.last_area() - area) < 25082.0:
            convex_size_q.insert((area,hull))
        else:
            if area < 0.5*convex_size_q.calc_mean():
                hull = convex_size_q.last_convex()
                
        cv2.drawContours(mask, [hull], -1, (255, 255, 255), cv2.FILLED)
        
        # invert mask so shapes are white on black background
        mask_inv = 255 - mask
        
        # create new (blue) background
        bckgnd = np.full_like(img, 255)

        # apply mask to image
        image_masked = cv2.bitwise_and(img, img, mask=mask)
        
        bckgnd = 255 - bckgnd

        # apply inverse mask to background
        bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
        

        # add together
        result = cv2.add(image_masked, bckgnd_masked)
    
    except:
        result = blurred
    
    img_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return img_bgr

import queue
import statistics
class Convex_size_queue:
    
    def __init__(self, number_of_items):
        self.queue = queue.Queue(number_of_items)
        self.queue_history = queue.Queue()

    def insert(self, item):
        if self.queue.full():
            self.queue.get()
        self.queue.put(item)
        self.queue_history.put(item)
        
    def calc_mean(self):
        convex_size_list = [list(self.queue.queue)[i][0] for i in range(len(list(self.queue.queue)))]
        if len(convex_size_list) == 0:
            return 0
        else:
            mean_size = statistics.mean(convex_size_list)
        return mean_size
    
    def last_convex(self):
        if len(list(self.queue.queue)) == 0:
            return 0
        return list(self.queue.queue)[-1][1]
    
    def last_area(self):
        if len(list(self.queue.queue)) == 0:
            return 0
        return list(self.queue.queue)[-1][0]
    
    def plot_area_history(self):
        convex_size_history_list = [list(self.queue_history.queue)[i][0] for i in range(len(list(self.queue_history.queue)))]
        
        max_slope = max([x - z for x, z in zip(convex_size_history_list[:-1], convex_size_history_list[1:])])
        print(max_slope)

        
        plt.plot(convex_size_history_list)
        plt.show()
    

convex_size_q = Convex_size_queue(5)

cap = cv2.VideoCapture(0)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))
    # cv2.imshow('flow HSV', draw_hsv(flow))
    cv2.imshow('contour', draw_contour_masked(gray, flow))

    key = cv2.waitKey(5)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()