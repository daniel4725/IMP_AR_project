import os
import sys

os.add_dll_directory(r'C:\opencv43\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\lib\gstreamer-1.0')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\lib')

import cv2
import numpy as np 

class Video_operations:
    
    def __init__(self):
        self.gstreamer_writer = None
        self.cap_receive = None
        self.frame_counter = 0
    
    def open_gstreamer_video_writer(self, IP: str = "192.168.0.169"):
        self.gstreamer_writer = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! rtph264pay ! udpsink host=' + IP + ' port=5005',cv2.CAP_GSTREAMER,0, 20, (1280,360), True)

        if not self.gstreamer_writer.isOpened():
            print('VideoWriter not opened')
            exit(0)
            
        return self.gstreamer_writer
    
    def close_gstreamer_video_writer(self):
        if self.gstreamer_writer is not None:
            self.gstreamer_writer.release()
    
    def open_gstreamer_video_capture(self):
        self.gstreamer_capture = cv2.VideoCapture('udpsrc port=5005 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
        
        if not self.gstreamer_capture.isOpened():
            print('VideoCapture not opened')
            exit(0)
        
        return self.gstreamer_capture
    
    def close_gstreamer_video_capture(self):
        if self.gstreamer_capture is not None:
            self.gstreamer_capture.release()
            
    def open_video_capture_from_path(self, path: str):
        self.video_capture_from_path = cv2.VideoWriter(path)

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
        
    def view_and_send_video(self, video_capture, video_writer, func):
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(fps)

        while True:
             
            ret, frame = video_capture.read()

            if not ret:
                print('empty frame')
                break
            
            self.frame_counter += 1
            if (self.frame_counter % 2) == 0:
                left_frame = self.get_left_image(frame)
                right_frame = self.get_right_image(frame)
                left_frame = func(left_frame)
                right_frame = func(right_frame)
                frame = self.image_concat(left_frame, right_frame)
            
            video_writer.write(frame)
            cv2.imshow('Video', frame)

            key = cv2.waitKey(int(500 / fps))
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
        