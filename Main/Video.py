import cv2
import numpy as np 
import threading
import time
from queue import Queue

class Video_operations:
    """
     This class gather video operations as open different kind of captures (PC, saved video and gstreamer), split video and more.
    """
    
    def __init__(self):
        """
        __init__ Initiate frame queue for determining the fps.
        """
        self.gstreamer_writer = None
        self.gstreamer_writer_2 = None
        self.cap_receive = None
        self.ready_to_read = True
        self.frame_queue = Queue()
        self.fps = None
        self.flip = False
        self.stereo = True
        self.finish_calibration = False
        self.sending_resolution = (1000, 320)
        self.count_down = 20
        self.image_shape = (320, 1280)
    
    def open_gstreamer_video_writer(self, IP: str = "192.168.0.169", IP_2: str = ''):
        """
        open_gstreamer_video_writer Open Gstreamer stream of x264 data.

        Args:
            IP (str, optional): The IP of the first device that data is been sending to. Defaults to "192.168.0.169".
            IP_2 (str, optional): The IP of the second to send. Defaults to '' (If it is empty, only the first stream will be create).

        Returns:
            _type_: _description_
        """
        self.gstreamer_writer = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast ! rtph264pay ! udpsink host=' + IP + ' port=5005',cv2.CAP_GSTREAMER,0, 20, self.sending_resolution, True)

        if IP_2 != '':
            self.gstreamer_writer_2 = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast key-int-max=12 cabac=1 bframes=2 ! rtph264pay ! udpsink host=' + IP_2 + ' port=5005',cv2.CAP_GSTREAMER,0, 20, self.sending_resolution, True)
        if not self.gstreamer_writer.isOpened():
            print('VideoWriter not opened')
            exit(0)
        return self.gstreamer_writer
    
    def close_gstreamer_video_writer(self):
        """
        close_gstreamer_video_writer Release cv2.writer object.
        """
        if self.gstreamer_writer is not None:
            self.gstreamer_writer.release()
        
        if self.gstreamer_writer_2 is not None:
            self.gstreamer_writer_2.release()

    def open_mp4_video_writer(self):
        """
        open_mp4_video_writer Create mp4 video writer.
        """
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.mp4_video_writer = cv2.VideoWriter("output_video.mp4", self._fourcc, 15.0, (1280, 320))
        if not self.mp4_video_writer.isOpened():
            print('VideoWriter is not opened')
            exit(0)

    def close_mp4_video_writer(self):
        """
        close_mp4_video_writer Close mp4 video writer.
        """
        if self.mp4_video_writer is not None:
            self.mp4_video_writer.release()
    
    def open_gstreamer_video_capture(self, flip: bool = False):
        """
        open_gstreamer_video_capture Read Gstreamer stream of x264 data in port 5005

        Args:
            flip (bool, optional): If the camera direction is front, this arg flip the video. Defaults to False.

        Returns:
            captrue (cv2 Video Capture): get frames from specific capture.
        """
        self.gstreamer_capture = cv2.VideoCapture('udpsrc port=5005 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        self.flip = flip
        if not self.gstreamer_capture.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        self.image_shape = (self.gstreamer_capture.read()[1]).shape
        return self.gstreamer_capture

    def close_gstreamer_video_capture(self):
        """
        close_gstreamer_video_capture close the capture.
        """
        if self.gstreamer_capture is not None:
            self.gstreamer_capture.release()
            
    def open_pc_video_capture(self, port_num: int, flip: bool = False, stereo: bool = True):
        """
        open_pc_video_capture Open capture in accordance to the chosen device in the computer.

        Args:
            port_num (int): Number of device.
            flip (bool, optional): If the camera direction is front, this arg flip the video. Defaults to False.
            stereo (bool, optional): Change to true if the device return image from two cameras. Defaults to True.

        Returns:
            captrue (cv2 Video Capture): get frames from specific capture.
        """
        self.pc_capture = cv2.VideoCapture(port_num)
        self.flip = flip
        self.stereo = stereo
        if not self.pc_capture.isOpened():
            print('VideoCapture not opened')
            exit(0)
            
        self.image_shape = (self.pc_capture.read()).shape
        
        return self.pc_capture
    
    def close_pc_video_capture(self):
        """
        close_pc_video_capture close the pc video capture
        """
        if self.pc_capture is not None:
            self.pc_capture.release()
            
    def close_video_capture_from_path(self):
        """
        close_video_capture_from_path close the video capture
        """
        if self.video_capture_from_path is not None:
            self.video_capture_from_path.release()
            
    def open_video_capture_from_path(self, path: str, flip: bool = False, stereo: bool = True):
        """
        open_video_capture_from_path Open video capture from save video.

        Args:
            path (str): The path of the saved video.
            flip (bool, optional): If the camera direction is front, this arg flip the video. Defaults to False.
            stereo (bool, optional): Change to true if the device return image from two cameras. Defaults to True.

        Returns:
            captrue (cv2 Video Capture): get frames from specific capture.
        """
        self.video_capture_from_path = cv2.VideoCapture(path)
        self.flip = flip
        self.stereo = stereo
        if not self.video_capture_from_path.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        self.image_shape = (self.video_capture_from_path.read()).shape
        return self.video_capture_from_path
        
    def start_thread_record_view_send(self, capture, func, write: bool = True, Save_video: bool = False):
        """
        start_thread_record_view_send This function starts two thread, record frames and process them.

        Args:
            capture (_type_): get frames from specific capture.
            func (_type_): function that is been processed in the process thread. 
            write (bool, optional): Check if to send the video to a gstreamer writer. Defaults to True.
            Save_video (bool, optional): Check if to save the video as mp4. Defaults to False.
        """
        crop_x = int((self.image_shape[1]/2) * 0.3)
        record_thread = threading.Thread(target=self.__thread_record_from_camera, args=(capture, Save_video))
        view_and_send_thread = threading.Thread(target=self.__view_thread_video_and_send_video, args=(func, crop_x , write, Save_video))
        record_thread.start()
        view_and_send_thread.start()
        record_thread.join()
        view_and_send_thread.join()

    def __thread_record_from_camera(self, captrue):
        """
        __thread_record_from_camera This function push frames into queue (that is thread safe), 
        only when the main thread is ready to receive a new frame, this way there is no latency. 

        Args:
            captrue (cv2 Video Capture): get frames from specific capture.
        """
        while True:
             
            ret, frame = captrue.read()
            self.fps = captrue.get(cv2.CAP_PROP_FPS)

            if not ret:
                print('empty frame')
                break

            if (self.ready_to_read):
                self.ready_to_read = False
                self.frame_queue.put(frame)
    
    def __view_thread_video_and_send_video(self, func, crop_x: int, write: bool = True, Save_video: bool = False):
        while True:

            frame = self.frame_queue.get()
            
            if self.stereo is True:
                left_frame = self.get_left_image(frame)
                right_frame = self.get_right_image(frame)
                if self.flip:
                    left_frame = cv2.flip(left_frame, 1)
                    right_frame = cv2.flip(right_frame, 1)
                frame = self.image_concat(left_frame, right_frame)
            else:
                if self.flip:
                    frame = cv2.flip(frame, 1)

            s = time.time()
            frame, self.finish_calibration = func(frame, crop_x)
            # print((time.time() - s))
            cv2.imshow("Before Reshape", frame)

            self.ready_to_read = True

            if Save_video is True:
                self.mp4_video_writer.write(frame)

            if write is True:
                frame = self.reshspe4phone(frame)
                self.gstreamer_writer.write(frame)
                if self.gstreamer_writer_2 is not None:
                    self.gstreamer_writer_2.write(frame)

            cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video', frame)
            # cv2.resizeWindow('Video', 1280*2, 320*2)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        
    def reshspe4phone(self, frame: np.array):
        """
        reshspe4phone This function resize stereo image with zero spaces for better fitting to AR phone preseting.

        Args:
            frame (np.array): _description_

        Returns:
            np.array: reshaped image.
        """
        crop_factor = 0.2
        left_frame = self.get_left_image(frame)
        right_frame = self.get_right_image(frame)
        roi = [round(left_frame.shape[0] * 0), round(left_frame.shape[0] * 1),
                    round(left_frame.shape[1] * crop_factor),
                    round(left_frame.shape[1] * (1 - crop_factor))]  # [y_start, y_end, x_start, x_end]
        crop_l = left_frame[roi[0]:roi[1], roi[2]:roi[3]]
        crop_r = right_frame[roi[0]:roi[1], roi[2]:roi[3]]
        mid_zeros = np.zeros((crop_l.shape[0], round(crop_l.shape[1] * 0.1 + 10), 3), dtype='uint8')
        side_zeros = np.zeros((crop_l.shape[0], round(crop_l.shape[1] * 0.25)+1 - 5, 3), dtype='uint8')
        reshaped = np.concatenate([side_zeros, crop_l, mid_zeros, crop_r, side_zeros], axis=1)
        return reshaped
    
    def get_left_image(self, image: np.array):
        """
        get_left_image return the left side of stereo image.

        Args:
            image (np.array): 3 channel image.

        Returns:
            np.array: 3 channel image.
        """
        return image[:, 0:int(image.shape[1] / 2), :]
    
    def get_right_image(self, image: np.array):
        """
        get_right_image return the right side of stereo image.

        Args:
            image (np.array): 3 channel image.

        Returns:
            np.array: 3 channel image.
        """
        return image[:, int(image.shape[1] / 2):, :]
    
    def image_concat(self, left_image: np.array, right_image: np.array):
        """
        image_concat concatenate two images together.

        Args:
            left_image (np.array): left image.
            right_image (np.array): right image

        Returns:
            np.array: concatenated image.
        """
        return np.concatenate([left_image, right_image], axis=1)
    
if __name__ == "__main__":
    
    video = Video_operations()
    
    def empty_func(image: np.array, crop_x: int):
        return (image, False)
    
    capture = video.open_gstreamer_video_capture(flip=False)
        
    video.start_thread_record_view_send(capture, empty_func, write=False, Save_video=False)
        
    video.close_gstreamer_video_capture()
        