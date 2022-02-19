import cv2
import threading
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import colors
import time
import pickle
import matplotlib.pyplot as plt
from scipy import ndimage
from Video import Video_operations
import os

from aux_functions import *
from table_handling import CornersFollower
from functools import wraps


def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time.time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time.time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it

class Calibration:
    def __init__(self, video_operations: Video_operations, stereo: bool = True):
        self.timer = 3
        self.count_down = 3
        self.timer_started = False
        self.timer_finished = False
        self.GMM_Image = np.zeros((400, 400))
        self.video_operations = video_operations
        self.roi = None
        self.GMM_Model = None
        self.components_number = 4
        self.hand_contour = None
        self.image_shape = None 
        self.capture_state = 0
        self.stereo = stereo
        self.crop_empty_frame_gray = None
        self.tol = 0
        self.timing_thread = None
        self.capture_image = None
        self.gmm_result_figure = None
        self.data = None
        self.label_to_color = {
            0: self.__color_to_vec("black"),
            1: self.__color_to_vec("tab:blue"),
            2: self.__color_to_vec("tab:orange"),
            3: self.__color_to_vec("tab:green"),
            4: self.__color_to_vec("tab:red"),
            5: self.__color_to_vec("tab:purple"),
            6: self.__color_to_vec("tab:brown"),
            7: self.__color_to_vec("tab:pink"),
        }
        self.calibration_state_names = {
            "look_at_table"      : 0,
            "calibrate_table"    : 1,
            "preview_corners"    : 2,
            "get_image_shape"    : 3,
            "capture_hands"      : 4,
            "gmm_train"          : 5,
            "preview_results"    : 6,
            "check_segmentation" : 7,
            "finish_calibration" : 8
        }
        self.calibrate_state = self.calibration_state_names["look_at_table"]
        self.__create_output_directory()
        
    def GMM_calibrate(self, image: np.array):
        return self.__State_machine(image, self.calibrate_state, self.stereo)
        
    def __State_machine(self, image: np.array, state: int, stereo: bool = False):
        
        if state == self.calibration_state_names["look_at_table"]:
            if self.timer_started is False:
                self.timing_thread = threading.Thread(target=self.__timer_sec)
                self.timing_thread.start()
            if self.timer_finished is True:
                self.timer_finished = False
                self.calibrate_state = self.calibration_state_names["calibrate_table"]
            return self.__look_at_table(image)
        elif state == self.calibration_state_names["calibrate_table"]:
            self.__calibrate_table(image, True)
            self.calibrate_state = self.calibration_state_names["preview_corners"]
            return self.__preview_corners(image)
        elif state == self.calibration_state_names["preview_corners"]:
            if self.timer_started is False:
                self.timer = 10
                self.timing_thread = threading.Thread(target=self.__timer_sec)
                self.timing_thread.start()
            if self.timer_finished is True:
                self.timer_finished = False
                self.calibrate_state = self.calibration_state_names["get_image_shape"]
            return self.corner_result_image
        elif state == self.calibration_state_names["get_image_shape"]:
            if stereo is True:
                self.__get_image_from_camera_shape(self.video_operations.get_left_image(image))
            else:
                self.__get_image_from_camera_shape(image)
            self.mask = self.__create_hand_mask(self.__create_hand_contour())
            self.sleeve_mask = self.__create_hand_mask(self.sleeve_contour)    
            self.calibrate_state = self.calibration_state_names["capture_hands"]
            return image
        elif state == self.calibration_state_names["capture_hands"]:
            return self.capture_hand(image, True, self.stereo)
        elif state == self.calibration_state_names["gmm_train"]:
            return self.gmm_train(self.GMM_Image, Save_model=True)
        elif state == self.calibration_state_names["preview_results"]:
            if self.timer_finished is True:
                self.timer_finished = False
                self.calibrate_state = self.calibration_state_names["check_segmentation"]
            return self.gmm_result_figure
        elif state == self.calibration_state_names["check_segmentation"]:
            return self.__check_if_segmentation_is_good(image)
        elif state == self.calibration_state_names["finish_calibration"]:
            return image
    
    def __create_output_directory(self):
        self.main_directory = os.path.dirname(os.path.abspath(__file__))
        self.output_directory_path = os.path.join(self.main_directory, "calibration_output")
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_output")) is False:
            os.mkdir(self.output_directory_path)
    
    def __look_at_table(self, image: np.array):
        self.video_operations.draw_text_two_image_or_one(image, "Look at the table please", (0, 50))
        self.video_operations.draw_text_two_image_or_one(image, f"{self.count_down}", (0, 100))
        return image

    def __preview_corners(self, image: np.array):
        im_l = self.video_operations.get_left_image(image)
        im_r = self.video_operations.get_right_image(image)
        for i, (p1, p2) in enumerate(zip(self.corner_follower_l.current_corners, np.roll(self.corner_follower_l.current_corners, shift=1, axis=0))):
            cv2.line(im_l, tuple(p1), tuple(p2), 255)
        for i, (p1, p2) in enumerate(zip(self.corner_follower_r.current_corners, np.roll(self.corner_follower_r.current_corners, shift=1, axis=0))):
            cv2.line(im_r, tuple(p1), tuple(p2), 255)
        self.corner_result_image = self.video_operations.image_concat(im_l, im_r)
        return self.corner_result_image
    
    @measure
    def __calibrate_table(self, image: np.array, Save_model:bool = False): 
        im_l = self.video_operations.get_left_image(image)
        im_r = self.video_operations.get_right_image(image)
        self.corner_follower_l = CornersFollower(im_l, show=True, name="c_follow_l")  # Create the corner follower for left image
        self.corner_follower_r = CornersFollower(im_r, show=True, name="c_follow_r")  # Create the corner follower for right image
        if (Save_model):
            # save the model to disk
            filename = 'corner_follower_l.sav'
            pickle.dump(self.corner_follower_l, open(os.path.join(self.output_directory_path, filename), 'wb'))
            filename = 'corner_follower_r.sav'
            pickle.dump(self.corner_follower_r, open(os.path.join(self.output_directory_path, filename), 'wb'))
    
    def __get_image_from_camera_shape(self, image: np.array):
        """
        __get_image_from_camera_shape Get the shape of the image from the camera

        Args:
            image (np.array): 3-channel image
        
        Returns:
            self.image_shape (np.array): 3 array.
        """
        self.image_shape = image.shape
        return self.image_shape
    
    def __create_hand_contour(self):
        """
        __create_hand_contour Create hand contour in specific place on image.

        Returns:
            list: cv2.contuor array with all points of the contour
        """
        zero_image = np.zeros((self.image_shape[0], self.image_shape[1])).astype(np.uint8)
        self.roi = [round(zero_image.shape[0] * 0.2), round(zero_image.shape[0] * 0.8), round(zero_image.shape[1] * 0.6), round(zero_image.shape[1] * 0.9)]  # [y_start, y_end, x_start, x_end]
        cropPrev = zero_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        crop_shape = cropPrev.shape
        handExample = cv2.imread('Full_hand.jpeg')
        handExample = cv2.resize(handExample, (crop_shape[1], crop_shape[0]))
        handExampleGray = cv2.cvtColor(handExample, cv2.COLOR_BGR2GRAY)
        zero_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = 255 - handExampleGray
        ret, mask = cv2.threshold(zero_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.hand_contour = contours[1]
        self.sleeve_contour = contours[0]
        return self.hand_contour
    
    def __create_hand_mask(self, contour):
        """
        __create_hand_mask Create mask of hand in the shape of full image from camera.

        """
        zero_image = np.zeros((self.image_shape[0], self.image_shape[1])).astype(np.uint8)
        self.video_operations.draw_contour_two_image_or_one(zero_image, contour, channels=1, stereo=False)
        
        img_fill_holes = ndimage.binary_fill_holes(zero_image).astype(np.uint8)
        mask = cv2.normalize(img_fill_holes, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return mask

    def __timer_sec(self):
        self.timer_started = True
        for i in range(self.timer):
            time.sleep(1)
            self.count_down -= 1
        self.timer_finished = True
        self.timer_started = False

    def __check_if_segmentation_is_good(self, image: np.array):
        self.video_operations.draw_text_two_image_or_one(image, "Is the segmentation good (see only palm)?", (0, 50))
        self.video_operations.draw_text_two_image_or_one(image, "Press Y/N", (0, 100))
        key = cv2.waitKey(30)
        if key == 121: # key == Y
            self.calibrate_state = self.calibration_state_names["finish_calibration"]
        elif key == 110: # key == N
            self.calibrate_state = self.calibration_state_names["capture_hands"]
        return self.__preview_calibrated_segmentation(image)
    
    def capture_hand(self, image: np.array, print_roi_match: bool = False, stereo: bool = False):
        
        if self.capture_state == 0:
            if stereo is True:
                capture_image = self.video_operations.get_left_image(image)
            else:
                capture_image = image
            crop_empty_frame = capture_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            self.crop_empty_frame_gray = cv2.cvtColor(crop_empty_frame, cv2.COLOR_BGR2GRAY)
            numPixels = self.crop_empty_frame_gray.shape[0] * self.crop_empty_frame_gray.shape[1]
            self.tol = numPixels / 10
            change = 0
            self.timer = 3
            self.count_down = 3
            self.timing_thread = threading.Thread(target=self.__timer_sec)
            self.capture_state = 1
        elif self.capture_state == 1:
            if stereo is True:
                self.capture_image = self.video_operations.get_left_image(image)
            else:
                self.capture_image = image
            crop_frame = self.capture_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            subImg = cv2.subtract(crop_frame_gray, self.crop_empty_frame_gray)
            y = subImg.reshape(1, -1)
            change = (y >= 10).sum()
            if(print_roi_match):
                print(self.tol)
                print(change)
            if change >= self.tol:
                self.capture_state = 2
                self.timing_thread.start()
            self.video_operations.draw_contour_two_image_or_one(image, self.hand_contour, 3, stereo)
            self.video_operations.draw_contour_two_image_or_one(image, self.sleeve_contour, 3, stereo)
        elif self.capture_state == 2:
            if self.count_down != 0:
                self.video_operations.draw_text_two_image_or_one(image, str(self.count_down), (0, 50), stereo)
                self.video_operations.draw_contour_two_image_or_one(image, self.hand_contour, 3, stereo)
                self.video_operations.draw_contour_two_image_or_one(image, self.sleeve_contour, 3, stereo)
            elif self.count_down == 0:
                if stereo is True:
                    self.GMM_Image = np.array(self.video_operations.get_left_image(image), copy=True)
                else:
                    self.GMM_Image = np.array(image, copy=True)
                self.video_operations.draw_text_two_image_or_one(image, "Image Saved", (0, 50), stereo)
                self.video_operations.draw_contour_two_image_or_one(image, self.hand_contour, 3, stereo)
                self.video_operations.draw_contour_two_image_or_one(image, self.sleeve_contour, 3, stereo)
                self.capture_state = 0
                self.calibrate_state = self.calibration_state_names["gmm_train"]
                self.timing_thread.join()
        return image
        
    def gmm_train(self, GMM_image: np.array, Save_model: bool = True):
        Shape = GMM_image.shape
        imageLAB = cv2.cvtColor(GMM_image, cv2.COLOR_BGR2LAB)

        L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()
        data = np.array([a, b]).transpose()

        n_components = self.components_number
        self.GMM_Model = GaussianMixture(n_components=n_components)
        self.GMM_Model.fit(data)
        GMM_Labels = self.GMM_Model.predict(data)
        
        segmented_labels = np.array(GMM_Labels).reshape(Shape[0], Shape[1]).astype(np.uint8)
        
        self.two_comp_label_list_hand = self.__get_most_valued_gmm_labels(segmented_labels, self.mask)
        self.two_comp_label_list_sleeve = self.__get_most_valued_gmm_labels(segmented_labels, self.sleeve_mask)
        segmented = self.__get_two_comp_segmentation(segmented_labels, self.two_comp_label_list_hand)

        if (Save_model):
            # save the model to disk
            filename = 'hand_gmm_model.sav'
            pickle.dump(self.GMM_Model, open(os.path.join(self.output_directory_path, filename), 'wb'))
            filename = 'hand_best_labels.sav'
            pickle.dump(self.two_comp_label_list_hand, open(os.path.join(self.output_directory_path, filename), 'wb'))
            filename = 'hand_sleeve_labels.sav'
            pickle.dump(self.two_comp_label_list_sleeve, open(os.path.join(self.output_directory_path, filename), 'wb'))

        self.__create_multiple_image((segmented_labels + 1), GMM_image, segmented)
        left_image = self.video_operations.image_resize(self.data, 0.7)
        right_image = self.video_operations.image_resize(self.data, 0.7)

        self.gmm_result_figure = self.video_operations.image_concat(left_image, right_image)
        self.calibrate_state = self.calibrate_state = self.calibration_state_names["preview_results"]
        self.timer_finished = False
        self.timer = 10
        self.timing_thread = threading.Thread(target=self.__timer_sec)
        self.timing_thread.start()
        
        return self.gmm_result_figure
        
    def __create_multiple_image(self, segmented_labels, GMM_image, segmented):    
        segmented_labels_image = self.__convert_lables_to_rgb_image(segmented_labels)
        two_lables_image = self.__convert_lables_to_rgb_image(segmented)
        first_line_image = np.concatenate([GMM_image, segmented_labels_image, two_lables_image], axis=1)
        crop_by_mask_labels_image = self.__convert_lables_to_rgb_image(self.crop_by_mask_segmented_labels)
        inv_crop_by_mask_labels_image = self.__convert_lables_to_rgb_image(self.invert_crop_by_mask_segmented_labels)
        second_line_image = np.concatenate([crop_by_mask_labels_image, inv_crop_by_mask_labels_image], axis=1)
        second_line_image = self.video_operations.three_dim_image_resize(second_line_image, first_line_image.shape)
        final_image = np.concatenate([first_line_image, second_line_image], axis=0)
        self.data = cv2.resize(final_image, (segmented_labels.shape[1],segmented_labels.shape[0]))
    
    def __print_lables_by_color_name(self, label_list: list):
        print("Choosen colors: ")
        for label in label_list:
            print((self.label_to_color[label + 1])[1])
    
    def __convert_lables_to_rgb_image(self, labled_image: np.array):
        h, w = labled_image.shape
        img_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for gray, rgb in self.label_to_color.items():
            img_rgb[labled_image == gray, :] = rgb[0]
        
        return img_rgb        
    
    def __color_to_vec(self, color: str):
        return (tuple([int(255*x) for x in colors.to_rgb(color)]), color)
        
    def __count_labels(self, prediction: np.array):
        unique, counts = np.unique(prediction, return_counts=True)
        label_dict = dict(zip(unique, counts))
        del label_dict[0]
        sorted_dic = {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1])}
        keys = list(sorted_dic.keys())
        label_list = [keys[-1]-1]
        for i in range(len(keys)-1):
            if sorted_dic[keys[i]] > (0.25)*sorted_dic[keys[-1]]:
                label_list.append(keys[i] - 1)
        return label_list
    
    def __get_most_valued_gmm_labels(self, n_comp_segmented_img: np.array, contour_mask):
        ret, mask = cv2.threshold(contour_mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img = n_comp_segmented_img + 1
        self.crop_by_mask_segmented_labels = cv2.bitwise_and(img, img, mask = mask)
        self.invert_crop_by_mask_segmented_labels = cv2.bitwise_and(img, img, mask = mask_inv)
        good_label_list = self.__count_labels(self.crop_by_mask_segmented_labels)
        bad_label_list = self.__count_labels(self.invert_crop_by_mask_segmented_labels)
        # self.two_comp_label_list =  self.__remove_bad_labels(good_label_list, bad_label_list)
        self.__print_lables_by_color_name(good_label_list)
        return good_label_list
        
    def __preview_calibrated_segmentation(self, image: np.array):
        Shape = image.shape
        imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()
        data = np.array([a, b]).transpose()
        
        GMM_Labels = self.GMM_Model.predict(data)
        
        segmented_labels = np.array(GMM_Labels).reshape(Shape[0], Shape[1]).astype(np.uint8)
        
        segmented = self.__get_two_comp_segmentation(segmented_labels, self.two_comp_label_list_hand)
        
        return self.__convert_lables_to_rgb_image(segmented)

    def __remove_bad_labels(self, main_list: list, second_list: list):
        if (len(main_list) == 1):
            return main_list
        else:
            for i in range(len(second_list)):
                try:
                    main_list.remove(second_list[i])
                except:
                    pass
            return main_list
    
    def __get_two_comp_segmentation(self, n_comp_segmented_img: np.array, two_comp_label_list):
        reduced_GMM_Labels_segmented_img = np.zeros(n_comp_segmented_img.shape)
        reduced_GMM_Labels_segmented_img[np.isin(n_comp_segmented_img, two_comp_label_list)] = 1
        return reduced_GMM_Labels_segmented_img
    
if __name__ == "__main__":
    
    video = Video_operations()
    cal = Calibration(video, True)
    
    # gstreamer_writer = video.open_gstreamer_video_writer("192.168.0.131", (1280, 320))
    capture = video.open_gstreamer_video_capture(flip=False)
    # capture = video.open_video_capture_from_path("regular_video.mp4", stereo=True)
    # capture = video.open_pc_video_capture(0, flip=True, stereo=False)
    video.start_thread_record_view_send(capture, cal.GMM_calibrate, write=False)
    # video.view_and_send_video(gstreamer_capture, gstreamer_writer, test_function)
    video.close_gstreamer_video_writer()
    video.close_gstreamer_video_capture()
    # video.close_pc_video_capture()
    # video.close_video_capture_from_path()

    
    

