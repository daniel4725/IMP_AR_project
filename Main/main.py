
from multiprocessing import Process

from Video import Video_operations
from Calibration import Calibration
from ClassAR import AR
import cv2
import numpy as np



class Main():
    def __init__(self):
        self.state = 2
    
    def main(self):
        self.video = Video_operations()

        self.video.open_mp4_video_writer()
        self.video.open_gstreamer_video_writer("192.168.0.169", "192.168.0.141")
        capture = self.video.open_gstreamer_video_capture(flip=False)
        # capture = video.open_pc_video_capture(1)
        
        self.video.start_thread_record_view_send(capture, self.main_state_machine, write=True, Save_video=False)
        
        self.video.close_gstreamer_video_writer()
        self.video.close_gstreamer_video_capture()
        self.video.close_mp4_video_writer()
        # video.close_pc_video_capture()    
    
    def main_state_machine(self, frame: np.array, crop_x: int):
        if self.state == 0:
            self.cal = Calibration(self.video, crop_x, True)
            self.state = 1
            return (frame, False)
        elif self.state == 1:
            frame, ret = self.cal.GMM_calibrate(frame, Save_model=True)
            if self.video.finish_calibration is True:
                self.state = 2
            return (frame, ret)
        elif self.state == 2:
            im_l = self.video.get_left_image(frame)
            im_r = self.video.get_right_image(frame)
            self.aug_real = AR(im_l, im_r, crop_x=crop_x)
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
    
    