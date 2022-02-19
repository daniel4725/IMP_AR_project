import os
import sys

os.add_dll_directory(r'C:\opencv43\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\lib\gstreamer-1.0')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\lib')

import cv2
import numpy as np 
import threading
import time
from queue import Queue

class Video_operations:
    
    def __init__(self):
        self.gstreamer_writer = None
        self.cap_receive = None
        self.ready_to_read = True
        self.frame_queue = Queue()
        self.fps = None
        self.flip = False
        self.stereo = True
        self.finish_calibration = False
    
    def open_gstreamer_video_writer(self, IP: str = "192.168.0.169", resolution: tuple = (1280,360)):
        self.gstreamer_writer = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! rtph264pay ! udpsink host=' + IP + ' port=5005',cv2.CAP_GSTREAMER,0, 20, resolution, True)

        if not self.gstreamer_writer.isOpened():
            print('VideoWriter not opened')
            exit(0)
            
        return self.gstreamer_writer
    
    def close_gstreamer_video_writer(self):
        if self.gstreamer_writer is not None:
            self.gstreamer_writer.release()
    
    def open_gstreamer_video_capture(self, flip: bool = False):
        self.gstreamer_capture = cv2.VideoCapture('udpsrc port=5005 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        self.flip = flip
        if not self.gstreamer_capture.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        return self.gstreamer_capture
    
    def close_gstreamer_video_capture(self):
        if self.gstreamer_capture is not None:
            self.gstreamer_capture.release()
            
    def open_pc_video_capture(self, port_num: int, flip: bool = False, stereo: bool = True):
        self.pc_capture = cv2.VideoCapture(port_num)
        self.flip = flip
        self.stereo = stereo
        if not self.pc_capture.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        return self.pc_capture
    
    def close_pc_video_capture(self, port_num: int, flip: bool = False):
        if self.pc_capture is not None:
            self.pc_capture.release()
            
    def close_video_capture_from_path(self, port_num: int, flip: bool = False):
        if self.video_capture_from_path is not None:
            self.video_capture_from_path.release()
            
    def open_video_capture_from_path(self, path: str, flip: bool = False, stereo: bool = True):
        self.video_capture_from_path = cv2.VideoCapture(path)
        self.flip = flip
        self.stereo = stereo
        if not self.video_capture_from_path.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        return self.video_capture_from_path
    
    def view_video(self, video_capture):
        
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(fps)

        while True:
             
            ret, frame = video_capture.read()

            if not ret:
                print('empty frame')
                break

            cv2.imshow('Video', frame)

            key = cv2.waitKey(int(500 / fps))
            if key == ord('q'):
                break
  
        cv2.destroyAllWindows()
        
    def start_thread_record_view_send(self, capture, func, write: bool = True):
        record_thread = threading.Thread(target=self.__thread_record_from_camera, args=(capture,))
        view_and_send_thread = threading.Thread(target=self.__view_thread_video_and_send_video, args=(func,write))
        record_thread.start()
        view_and_send_thread.start()
        record_thread.join()
        view_and_send_thread.join()
        
    def __thread_record_from_camera(self , captrue):
        while True:
             
            ret, frame = captrue.read()
            self.fps = captrue.get(cv2.CAP_PROP_FPS)

            if not ret:
                print('empty frame')
                break

            if (self.ready_to_read):
                self.ready_to_read = False
                self.frame_queue.put(frame)
    
    def __view_thread_video_and_send_video(self,func, write: bool = True):
        
        while True:
             
            frame = self.frame_queue.get()
            
            if self.stereo == True:
                left_frame = self.get_left_image(frame)
                right_frame = self.get_right_image(frame)
                if self.flip:
                    left_frame = cv2.flip(left_frame, 1)
                    right_frame = cv2.flip(right_frame, 1)
                frame = self.image_concat(left_frame, right_frame)
            else:
                if self.flip:
                    frame = cv2.flip(frame, 1)
                    
            frame , self.finish_calibration = func(frame)
            
            self.ready_to_read = True
            if write is True:
                self.gstreamer_writer.write(frame)
            cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video', frame)
            cv2.resizeWindow('Video', 1280*2, 320*2)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
  
        cv2.destroyAllWindows()
      
    def save_and_preview_video_from_other_video(self, func, source: str, destination: str, ):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(destination, -1, 20.0, (640,480))
        # get total number of frames
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        for image in range(int(totalFrames)):

            suc, img = cap.read()
               
            func_img = func(img)
            dim = (640, 480)
            func_img_resized = cv2.resize(func_img, dim, interpolation = cv2.INTER_AREA)
            imagenorm = cv2.normalize(func_img_resized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            out.write(imagenorm)
            cv2.imshow('Video', func_img)
            cv2.imshow('Original', img)

            key = cv2.waitKey(100)
            if key == ord('q'):
                break

            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    def draw_contour_two_image_or_one(self, image: np.array, contour: np.array, channels: int = 3, stereo: bool = False):
        if stereo == True:
            left_image = self.__draw_contour(self.get_left_image(image), contour, channels)
            right_image = self.__draw_contour(self.get_right_image(image), contour, channels)
            return self.image_concat(left_image, right_image)
        
        else:
            return self.__draw_contour(image, contour, channels)
        
    def draw_text_two_image_or_one(self, image: np.array, text: str, position: tuple, stereo: bool = False):
        if stereo == True:
            
            left_image = cv2.putText(self.get_left_image(image), f'{text}', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            right_image = cv2.putText(self.get_right_image(image), f'{text}', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            return self.image_concat(left_image, right_image)
        
        else:
            return cv2.putText(image, f'{text}', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    def __draw_contour(self, image: np.array, contour: np.array, channels: int = 3):
        """
        __draw_contour Draw contour on the required image.

        Args:
            image (np.array): Image.
            contour (np.array): cv2.contuor array with all points of the contour
            channels (int): number of color channels of the image.
        """

        if channels == 3:
            cv2.drawContours(image, contour, -1, (255,0,0), 3)
        elif channels == 1:
            cv2.drawContours(image, contour, -1, 255, 3)
        return image
    
    def get_left_image(self, image: np.array):
        return image[:, 0:int(image.shape[1] / 2), :]
    
    def get_right_image(self, image: np.array):
        return image[:, int(image.shape[1] / 2):, :]
    
    def image_concat(self, left_image: np.array, right_image: np.array):
        return np.concatenate([left_image, right_image], axis=1)
    
    def add_image_distorter(self, image: np.array):
        left_image = self.get_left_image(image)
        right_image = self.get_right_image(image)
        
        width = left_image.shape[1]
        height = left_image.shape[0]
        distCoeff = np.zeros((4, 1), np.float64)
        # TODO: add your coefficients here!
        k1 = 1.0e-4  # negative to remove barrel distortion
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0
        distCoeff[0, 0] = k1
        distCoeff[1, 0] = k2
        distCoeff[2, 0] = p1
        distCoeff[3, 0] = p2
        # assume unit matrix for camera
        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = width / 2.0  # define center x
        cam[1, 2] = height / 2.0  # define center y
        cam[0, 0] = 10.  # define focal length x
        cam[1, 1] = 10.  # define focal length y

        self.distCoeff = distCoeff
        self.cam = cam
        
        return self.__distort_and_concat(left_image, right_image)
        
    def __distort_and_concat(self, im_l, im_r):
        dst_l = cv2.undistort(im_l, self.cam, self.distCoeff)
        dst_r = cv2.undistort(im_r, self.cam, self.distCoeff)
        return np.concatenate([dst_l, dst_r], axis=1)
    
    def image_resize(self, image: np.array, factor: int):
        shape = image.shape
        new_image = np.zeros(shape).astype(np.uint8)
        hh, ww = shape[0] , shape[1] 

        if len(shape) == 3:
            resized_image_r = cv2.resize(image[:,:,0], (round(shape[1]*factor), round(shape[0]*factor)))
            resized_image_g = cv2.resize(image[:,:,1], (round(shape[1]*factor), round(shape[0]*factor)))
            resized_image_b = cv2.resize(image[:,:,2], (round(shape[1]*factor), round(shape[0]*factor)))
        else:
            resized_image = cv2.resize(image, (round(shape[1]*factor), round(shape[0]*factor)))
        h, w = round(shape[0]*factor) , round(shape[1]*factor)

        yoff = round((hh-h)/2)
        xoff = round((ww-w)/2)
        
        if len(shape) == 3:
            new_image[yoff:yoff+h, xoff:xoff+w, 2] = resized_image_r
            new_image[yoff:yoff+h, xoff:xoff+w, 1] = resized_image_g
            new_image[yoff:yoff+h, xoff:xoff+w, 0] = resized_image_b
        else:
            new_image[yoff:yoff+h, xoff:xoff+w] = resized_image
        return new_image
    
    def three_dim_image_resize(self, image: np.array, shape: np.array):
        new_image = np.zeros(shape).astype(np.uint8)
        
        resized_image_r = cv2.resize(image[:,:,0], (shape[1], shape[0]))
        resized_image_g = cv2.resize(image[:,:,1], (shape[1], shape[0]))
        resized_image_b = cv2.resize(image[:,:,2], (shape[1], shape[0]))
        
        new_image[:,:,0] = resized_image_r
        new_image[:,:,1] = resized_image_g
        new_image[:,:,2] = resized_image_b
        
        return new_image    
if __name__ == "__main__":
    
    video = Video_operations()
    # video.view_video_from_path("regular_video_right.mp4")

    # from Calibration import Calibration
    
    # cal = Calibration()
    # cal.load_saved_model('hand_gmm_model.sav')
    # cal.load_saved_best_labels('hand_best_labels.sav')
    
    # from HandOperations import HandOperations
    
    # def get_hand_mask(image):
    #     hand = HandOperations(image=image)
    #     masked_image = hand.get_hand_mask(cal.get_segmentation(image))
    #     # count_image = hand.finger_count(masked_image)
    #     return masked_image
    
    # video.save_and_preview_video_from_other_video(get_hand_mask, "regular_video_left.mp4", "masked_regular_video_left.mp4")
    
    cap = video.open_gstreamer_video_capture()
    ret, image = cap.read()
    # video.view_video(cap)
    # video.close_gstreamer_video_writer()
        