
from multiprocessing import Process

from Video import Video_operations
from Calibration import Calibration
from ClassAR import AR
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



class Main():
    def __init__(self):
        self.state = 2
    
    def main(self):
        self.video = Video_operations()
        
        gstreamer_writer = self.video.open_gstreamer_video_writer("192.168.0.169")
        capture = self.video.open_gstreamer_video_capture(flip=False)
        # capture = video.open_pc_video_capture(1)
        self.video.start_thread_record_view_send(capture, self.main_state_machine, True)
        # video.view_and_send_video(gstreamer_capture, gstreamer_writer, test_function)
        self.video.close_gstreamer_video_writer()
        self.video.close_gstreamer_video_capture()
        # video.close_pc_video_capture()    
    
    def main_state_machine(self, frame: np.array):
        if self.state == 0:
            self.cal = Calibration(self.video, True)
            self.state = 1
            return (frame, False)
        elif self.state == 1:
            frame , ret = self.cal.GMM_calibrate(frame)
            if self.video.finish_calibration is True:
                self.state = 2
            return (frame, ret)
        elif self.state == 2:
            im_l = self.video.get_left_image(frame)
            im_r = self.video.get_right_image(frame)
            self.aug_real = AR(im_l, im_r)
            self.state = 3
            return (frame, False)
        elif self.state == 3:
            im_l = self.video.get_left_image(frame)
            im_r = self.video.get_right_image(frame)
            im_l, im_r = self.aug_real.get_AR(im_l, im_r)
            return (self.video.image_concat(im_l, im_r), False)
    
if __name__ == "__main__":
    main = Main()
    main.main()
    
    