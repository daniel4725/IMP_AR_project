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

class Calibration:
    """
     This class preform full calibration of the AR system. The calibration is most for getting hand and sleeve segmentation for the main app.
     The segmentation is with Gaussian Mixture Model.
    """
    
    def __init__(self, video_operations: Video_operations ,crop_x: int ,stereo: bool = True):
        """
        __init__ Init calibration instance.

        Args:
            video_operations (Video_operations): Get Video operation instance.
            crop_x (int): Amount of cropping of the image.
            stereo (bool, optional): Check if the image is stereo image. Defaults to True.
        """
        self.timer = 3
        self.count_down = 3
        self.timer_started = False
        self.timer_finished = False
        self.GMM_Image = np.zeros((400, 400))
        self.video_operations = video_operations
        self.roi = None
        self.GMM_Model = None
        self.components_number = 3
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
            1: self.__color_to_vec("tab:purple"),
            2: self.__color_to_vec("tab:orange"),
            3: self.__color_to_vec("tab:green"),
            4: self.__color_to_vec("tab:red"),
            5: self.__color_to_vec("tab:blue"),
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
        self.calibrate_state = self.calibration_state_names["get_image_shape"]
        self.__create_output_directory()
        self.finish_calibration = False
        
        self.LOC_TITLE = np.array([70 + crop_x, 20])
        self.LOC1 = np.array([crop_x, 20 + 60])
        self.LOC2 = np.array([crop_x, 40 + 60])
        self.LOC3 = np.array([crop_x, 60 + 60])
        self.LOC4 = np.array([crop_x, 80 + 60])
        self.LOC5 = np.array([crop_x, 100 + 60])
        
        self.offset = 50
        
        self.FONT_COLOR = (246, 158, 70)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.size = 0.7
        self.thick = 1
        
    def GMM_calibrate(self, image: np.array, Save_model: bool = False):
        """
        GMM_calibrate Start the calibration by calling to state machine.

        Args:
            image (np.array): 3 channel image
            Save_model (bool, optional): Check if to save the trained model. Defaults to False.

        Returns:
            np.array: image.
        """
        return (self.__State_machine(image, self.calibrate_state, self.stereo, Save_model), self.finish_calibration)
        
    def __State_machine(self, image: np.array, state: int, stereo: bool = False, Save_model: bool = False):
        """
        __State_machine Change states in calibration.

        Args:
            image (np.array): regular image.
            state (int): The current state.
            stereo (bool, optional): Check if image is stereo image. Defaults to False.
            Save_model (bool, optional): Check if to save the trained model. Defaults to False.

        Returns:
           np.array: image.
        """
        
        if state == self.calibration_state_names["look_at_table"]:
            if self.timer_started is False:
                self.timer = 1
                self.count_down = 1
                self.timing_thread = threading.Thread(target=self.__timer_sec)
                self.timing_thread.start()
            if self.timer_finished is True:
                self.timer_finished = False
                self.timing_thread.join()
                self.calibrate_state = self.calibration_state_names["calibrate_table"]
            return self.__look_at_table(image)
        elif state == self.calibration_state_names["calibrate_table"]:
            self.__calibrate_table(image, True)
            self.calibrate_state = self.calibration_state_names["preview_corners"]
            return self.__preview_corners(image)
        elif state == self.calibration_state_names["preview_corners"]:
            if self.timer_started is False:
                self.timer = 1
                self.timing_thread = threading.Thread(target=self.__timer_sec)
                self.timing_thread.start()
            if self.timer_finished is True:
                self.timer_finished = False
                self.timing_thread.join()
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
            image = self.capture_hand(image, True, self.stereo)
            image[:, :image.shape[1]//2] = 0
            return image
        elif state == self.calibration_state_names["gmm_train"]:
            self.gmm_train(self.GMM_Image, Save_model)
            return image
        elif state == self.calibration_state_names["preview_results"]:
            if self.timer_started is False:
                self.timer = 5
                self.timing_thread = threading.Thread(target=self.__timer_sec)
                self.timing_thread.start()
            if self.timer_finished is True:
                self.timer_finished = False
                self.timing_thread.join()
                self.calibrate_state = self.calibration_state_names["check_segmentation"]
            return image
        elif state == self.calibration_state_names["check_segmentation"]:
            return self.__check_if_segmentation_is_good(image)
        elif state == self.calibration_state_names["finish_calibration"]:
            self.finish_calibration = True
            return image
    
    def __create_output_directory(self):
        """
        __create_output_directory Create output directory with the GMM model and chosen lables for hand and sleeve.
        """
        self.main_directory = os.path.dirname(os.path.abspath(__file__))
        self.output_directory_path = os.path.join(self.main_directory, "calibration_output")
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_output")) is False:
            os.mkdir(self.output_directory_path)
    
    def __look_at_table(self, image: np.array):
        """
        __look_at_table Draw on the image title and count down.

        Args:
            image (np.array): _description_

        Returns:
            np.array: image.
        """
        self.draw_text_two_image(image, "Look at the table please", self.LOC_TITLE)
        self.draw_text_two_image(image, self.count_down, self.LOC1)
        return image

    def draw_text_two_image(self, image: np.array, text: str, position: np.array):
        """
        draw_text_two_image Draw on the stereo image text with offset for AR.

        Args:
            image (np.array): Regular image.
            text (str): Required text to draw.
            position (np.array): Position on the image.

        Returns:
            np.array: image with text.
        """
        left_position = (position[0] + self.offset, position[1])
        left_image = cv2.putText(self.video_operations.get_left_image(image), f'{text}', left_position, self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
        right_image = cv2.putText(self.video_operations.get_right_image(image), f'{text}', tuple(position), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
        return self.video_operations.image_concat(left_image, right_image)

    def draw_contour_two_image(self, image: np.array, contour: np.array):
        """
        draw_contour_two_image Draw on the stereo image contour with offset for AR.

        Args:
            image (np.array): Regular image.
            contour (np.array): The required shape.

        Returns:
            np.array: image with contour.
        """
        left_image = self.video_operations.get_left_image(image)
        right_image = cv2.drawContours(self.video_operations.get_right_image(image), contour, -1, self.FONT_COLOR, self.thick)
        return self.video_operations.image_concat(left_image, right_image)
    
    def __preview_corners(self, image: np.array):
        """
        __preview_corners Preview table segmentation on the image. 

        Args:
            image (np.array): Regular Image.

        Returns:
            np.array: image with corners mark.
        """
        im_l = self.video_operations.get_left_image(image)
        im_r = self.video_operations.get_right_image(image)
        for i, (p1, p2) in enumerate(zip(self.corner_follower_l.current_corners, np.roll(self.corner_follower_l.current_corners, shift=1, axis=0))):
            cv2.line(im_l, tuple(p1), tuple(p2), 255)
        for i, (p1, p2) in enumerate(zip(self.corner_follower_r.current_corners, np.roll(self.corner_follower_r.current_corners, shift=1, axis=0))):
            cv2.line(im_r, tuple(p1), tuple(p2), 255)
        self.corner_result_image = self.video_operations.image_concat(im_l, im_r)
        return self.corner_result_image
    
    def __calibrate_table(self, image: np.array, Save_model:bool = False): 
        """
        __calibrate_table Get table segmentation.

        Args:
            image (np.array): Regular Image.
            Save_model (bool, optional): Check if to save the trained model. Defaults to False.
        """
        im_l = self.video_operations.get_left_image(image)
        im_r = self.video_operations.get_right_image(image)
        self.corner_follower_l = CornersFollower(im_l, show=False, name="c_follow_l")  # Create the corner follower for left image
        self.corner_follower_r = CornersFollower(im_r, show=False, name="c_follow_r")  # Create the corner follower for right image
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
        self.roi = [round(zero_image.shape[0] * 0.2), round(zero_image.shape[0] * 0.8), round(zero_image.shape[1] * 0.55), round(zero_image.shape[1] * 0.75)]  # [y_start, y_end, x_start, x_end]
        cropPrev = zero_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        crop_shape = cropPrev.shape
        handExample = cv2.imread(os.path.join(self.main_directory, 'Full_hand.jpeg'))
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
        """
        __timer_sec Timer
        """
        self.timer_started = True
        for i in range(self.timer):
            time.sleep(1)
            self.count_down -= 1
        self.timer_finished = True
        self.timer_started = False

    def __check_if_segmentation_is_good(self, image: np.array):
        """
        __check_if_segmentation_is_good This state draw on the image a query if the hand and sleeve segmentation is good.

        Args:
            image (np.array): Regular image.

        Returns:
            np.array: segmented image.
        """
        image = self.__preview_calibrated_segmentation(image)
        self.draw_text_two_image(image, "Is the segmentation good?", self.LOC1)
        self.draw_text_two_image(image, "(see only hand palm)?", self.LOC2)
        self.draw_text_two_image(image, "Press Y/N", self.LOC3)
        self.draw_text_two_image(image, f"{self.hand_colors_strings}", self.LOC4)
        self.draw_text_two_image(image, f"{self.sleeve_colors_strings}" , self.LOC5)
        key = cv2.waitKey(20)
        if key == 121: # key == Y
            self.calibrate_state = self.calibration_state_names["finish_calibration"]
        elif key == 110: # key == N
            self.calibrate_state = self.calibration_state_names["capture_hands"]
        return image
    
    def capture_hand(self, image: np.array, print_roi_match: bool = False, stereo: bool = False):
        """
        capture_hand State machine to capture the image with hand getting inside an roi and then countdown until taking the image.

        Args:
            image (np.array): Regular image
            print_roi_match (bool, optional): Check if to print the match between the hand and the ROI. Defaults to False.
            stereo (bool, optional): Check if the image is stereo image. Defaults to False.

        Returns:
            np.array: image with text and contour.
        """
        
        if self.capture_state == 0:
            if stereo is True:
                capture_image = self.video_operations.get_right_image(image)
            else:
                capture_image = image
            crop_empty_frame = capture_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            self.crop_empty_frame_gray = cv2.cvtColor(crop_empty_frame, cv2.COLOR_BGR2GRAY)
            numPixels = self.crop_empty_frame_gray.shape[0] * self.crop_empty_frame_gray.shape[1]
            self.tol = numPixels / 10
            change = 0
            self.timer = 10
            self.count_down = 10
            self.timing_thread = threading.Thread(target=self.__timer_sec)
            self.capture_state = 1
        elif self.capture_state == 1:
            if stereo is True:
                self.capture_image = self.video_operations.get_right_image(image)
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
            self.draw_contour_two_image(image, self.hand_contour)
            self.draw_contour_two_image(image, self.sleeve_contour)
            self.draw_text_two_image(image, "Put your hand inside", self.LOC1)
            self.draw_text_two_image(image, "the hand contour", self.LOC2)

        elif self.capture_state == 2:
            if self.count_down != 0:
                self.draw_text_two_image(image, "Put your hand inside", self.LOC1)
                self.draw_text_two_image(image, "the hand contour", self.LOC2)
                self.draw_text_two_image(image, str(self.count_down), self.LOC3)
                self.draw_contour_two_image(image, self.hand_contour)
                self.draw_contour_two_image(image, self.sleeve_contour)
            elif self.count_down == 0:
                if stereo is True:
                    self.GMM_Image = np.array(self.video_operations.get_right_image(image), copy=True)
                else:
                    self.GMM_Image = np.array(image, copy=True)
                self.draw_text_two_image(image, "Image Saved", self.LOC1)
                self.capture_state = 0
                self.calibrate_state = self.calibration_state_names["gmm_train"]
                self.timing_thread.join()
        return image
        
    def gmm_train(self, GMM_image: np.array, Save_model: bool = True):
        """
        gmm_train The GMM train process.

        Args:
            GMM_image (np.array): Image with hand inside the ROI.
            Save_model (bool, optional): Check if to save the trained model. Defaults to True.

        Returns:
            _type_: _description_
        """
        Shape = GMM_image.shape
        imageLAB = cv2.cvtColor(GMM_image, cv2.COLOR_BGR2LAB)
        # imageLAB = GMM_image

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

        self.hand_colors_strings = "Hand: " + ", ".join(list(map(lambda label: (self.label_to_color[label])[1].replace('tab:', ''), self.two_comp_label_list_hand)))
        self.sleeve_colors_strings = "Sleeve: " + ", ".join(list(map(lambda label: (self.label_to_color[label])[1].replace('tab:', ''), self.two_comp_label_list_sleeve)))

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
        self.calibrate_state = self.calibration_state_names["check_segmentation"]
        
        return self.gmm_result_figure
        
    def __create_multiple_image(self, segmented_labels: np.array, GMM_image: np.array, segmented: np.array):
        """
        __create_multiple_image Create an image with multiple small images inside.

        Args:
            segmented_labels (np.array): _description_
            GMM_image (np.array): Regular image.
            segmented (np.array): Segmented image.
        """
        segmented_labels_image = self.__convert_lables_to_rgb_image(segmented_labels)
        two_lables_image = self.__convert_lables_to_rgb_image(segmented)
        first_line_image = np.concatenate([GMM_image, segmented_labels_image, two_lables_image], axis=1)
        crop_by_mask_labels_image = self.__convert_lables_to_rgb_image(self.crop_by_mask_segmented_labels)
        inv_crop_by_mask_labels_image = self.__convert_lables_to_rgb_image(self.invert_crop_by_mask_segmented_labels)
        second_line_image = np.concatenate([crop_by_mask_labels_image, inv_crop_by_mask_labels_image], axis=1)
        second_line_image = self.video_operations.three_dim_image_resize(second_line_image, first_line_image.shape)
        final_image = np.concatenate([first_line_image, second_line_image], axis=0)
        self.data = cv2.resize(final_image, (segmented_labels.shape[1], segmented_labels.shape[0]))
    
    def __print_lables_by_color_name(self, label_list: list):
        """
        __print_lables_by_color_name Print the color of each label.

        Args:
            label_list (list): label list.
        """
        print("Choosen colors: ")
        for label in label_list:
            print((self.label_to_color[label + 1])[1])
    
    def __convert_lables_to_rgb_image(self, labled_image: np.array):
        """
        __convert_lables_to_rgb_image Convert each labled pixel to 3 RGB pixel.

        Args:
            labled_image (np.array): labled_image

        Returns:
            np.array: RGB image.
        """
        h, w = labled_image.shape
        img_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for gray, rgb in self.label_to_color.items():
            img_rgb[labled_image == gray, :] = rgb[0]

        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    def __color_to_vec(self, color: str):
        """
        __color_to_vec Convert color name to RGB tuple with 0-255 values.

        Args:
            color (str): Color name.

        Returns:
            tuple: RGB 3-cjannels tuple.
        """
        return (tuple([int(255*x) for x in colors.to_rgb(color)]), color)
        
    def __count_labels(self, prediction: np.array):
        """
        __count_labels Count all values (lables) and select the most counted label and at least 25% of counts of the most counted label.

        Args:
            prediction (np.array): segmented image with labels.

        Returns:
            List: chosen labels.
        """
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
        """
        __get_most_valued_gmm_labels Check what are the most counted labels inside the contour.

        Args:
            n_comp_segmented_img (np.array): labled image.
            contour_mask (contour): contour.

        Returns:
            list: chosen label list.
        """
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
        """
        __preview_calibrated_segmentation Predict segmentation in image using trained GMM model after resizing to smaller image to speedup the process.

        Args:
            image (np.array): regular image.

        Returns:
            np.array: rgb image of the segmented image.
        """
        
        smaller_image = cv2.resize(image, ((image.shape[1] // 2), (image.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        Shape = smaller_image.shape
        imageLAB = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2LAB)
        # imageLAB = smaller_image

        L = np.array(imageLAB[:, :, 0]).flatten()
        a = np.array(imageLAB[:, :, 1]).flatten()
        b = np.array(imageLAB[:, :, 2]).flatten()
        data = np.array([a, b]).transpose()
        
        GMM_Labels = self.GMM_Model.predict(data)
        
        segmented_labels = np.array(GMM_Labels).reshape(Shape[0], Shape[1]).astype(np.uint8)
        
        segmented = self.__get_two_comp_segmentation(segmented_labels, self.two_comp_label_list_hand)
        
        rgb_image = self.__convert_lables_to_rgb_image(segmented_labels)
        
        return cv2.resize(rgb_image, ((image.shape[1]), (image.shape[0])), interpolation=cv2.INTER_AREA)
    
    def __get_two_comp_segmentation(self, n_comp_segmented_img: np.array, two_comp_label_list: list):
        """
        __get_two_comp_segmentation This function create new image with only two labels of main intrest and background.

        Args:
            n_comp_segmented_img (np.array): Labled image from GMM prediction.
            two_comp_label_list (list): List with all the lables of the main interst.

        Returns:
            np.array: Image with the labels main intrest and background.
        """
        reduced_GMM_Labels_segmented_img = np.zeros(n_comp_segmented_img.shape)
        reduced_GMM_Labels_segmented_img[np.isin(n_comp_segmented_img, two_comp_label_list)] = 1
        return reduced_GMM_Labels_segmented_img
    
if __name__ == "__main__":
    
    video = Video_operations()
    cal = Calibration(video, True)
    
    capture = video.open_gstreamer_video_capture(flip=False)
    # capture = video.open_video_capture_from_path("regular_video.mp4", stereo=True)
    # capture = video.open_pc_video_capture(0, flip=True, stereo=False)
    video.start_thread_record_view_send(capture, cal.GMM_calibrate, write=False)
    # video.view_and_send_video(gstreamer_capture, gstreamer_writer, test_function)
    video.close_gstreamer_video_writer()
    video.close_gstreamer_video_capture()
    # video.close_pc_video_capture()
    # video.close_video_capture_from_path()