
from multiprocessing import Process

from Video import Video_operations
import cv2
import numpy as np

from functools import wraps
from time import time

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it

def hand_contour(image: np.array):
        zero_image = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
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
    cv2.drawContours(image, contour, -1, (255,0,0), 3)
    

def test_function(frame):
    # import cv2
    # cv2.putText(frame, "test", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # contour = hand_contour(frame)
    # draw_contour(frame, contour)
    return frame


def main():
    video = Video_operations()
    
    gstreamer_writer = video.open_gstreamer_video_writer("192.168.0.131", (1280, 320))
    capture = video.open_gstreamer_video_capture(flip=False)
    # capture = video.open_pc_video_capture(1)
    video.start_thread_record_view_send(capture, test_function, True)
    # video.view_and_send_video(gstreamer_capture, gstreamer_writer, test_function)
    video.close_gstreamer_video_writer()
    # video.close_gstreamer_video_capture()
    # video.close_pc_video_capture()
    
    
    
if __name__ == "__main__":
    main()
    
    