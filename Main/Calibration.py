import cv2
import threading
import numpy as np
from sklearn.mixture import GaussianMixture
import time
import pickle
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from Video import Video_operations

class Calibration:
    def __init__(self, video_operations: Video_operations):
        self.timer = 10
        self.count_down = 10
        self.GMM_Image = np.zeros((400, 400))
        self.video_operations = video_operations
        self.roi = None
        self.GMM_Model = None
        self.components_number = 4
        self.hand_contour = None
        self.image_shape = None
        self.calibrate_state = 0
        self.capture_state = 0
        

    def GMM_calibrate(self, image: np.array):
        self.__State_machine(image, self.calibrate_state)
        
    
    def __State_machine(self, image: np.array, state: int):
        if state == 0:
            self.__get_image_from_camera_shape(image)
            self.__create_hand_mask()    
            self.calibrate_state = 1
            return image
        elif state == 1:
            pass

    
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
        self.roi = [zero_image.shape[0] - 340, zero_image.shape[0] - 120, zero_image.shape[1] - 240, zero_image.shape[1] - 20]  # [y_start, y_end, x_start, x_end]
        cropPrev = zero_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        crop_shape = cropPrev.shape
        handExample = cv2.imread('handExample.jpeg')
        handExample = cv2.resize(handExample, (crop_shape[1], crop_shape[0]))
        handExampleGray = cv2.cvtColor(handExample, cv2.COLOR_BGR2GRAY)
        zero_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = 255 - handExampleGray
        ret, mask = cv2.threshold(zero_image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.hand_contour = contours[0]
        return self.hand_contour
    
    
    def __create_hand_mask(self):
        """
        __create_hand_mask Create mask of hand in the shape of full image from camera.

        """
        zero_image = np.zeros((self.image_shape[0], self.image_shape[1])).astype(np.uint8)
        self.video_operations.draw_contour_two_image_or_one(zero_image, self.__create_hand_contour(), channels=1)

        img_fill_holes = ndimage.binary_fill_holes(zero_image).astype(np.uint8)
        self.mask = cv2.normalize(img_fill_holes, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return self.mask

    def timer_sec(self):
        for i in range(self.timer):
            time.sleep(1)
            self.count_down -= 1

    # def capture_hand(self, image: np.array, print_roi_match: bool = False):
        
    #     if self.capture_state == 0:
    #         crop_empty_frame = image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
    #         crop_empty_frame_gray = cv2.cvtColor(crop_empty_frame, cv2.COLOR_BGR2GRAY)
    #         numPixels = crop_empty_frame_gray.shape[0] * crop_empty_frame_gray.shape[1]
    #         tol = numPixels / 10
    #         change = 0
    #         startTimer = False
    #         t1 = threading.Thread(target=self.timer_sec)
    #         started_flag = False
    #         self.capture_state = 1
    #     elif self.capture_state == 1:
    #         crop_frame = image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
    #         crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    #         subImg = cv2.subtract(crop_frame_gray, crop_empty_frame_gray)
    #         y = subImg.reshape(1, -1)
    #         change = (y >= 10).sum()
    #         if(print_roi_match):
    #             print(tol)
    #             print(change)
    #         if change >= tol:
    #             startTimer = True
    #         if startTimer == True:
    #             t1.start()
    #             started_flag = True
    #         if self.flag != 0 and started_flag:
    #             cv2.putText(imgView, f'{self.flag}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             cv2.imshow('image', imgView)
    #             if self.video_writer is not None:
    #                 cv2.putText(left_image_view, f'{self.flag}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 cv2.putText(right_image_view, f'{self.flag}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
    #         elif self.flag == 0 and started_flag:
    #             cv2.putText(imgView, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #             self.GMM_Image = img
    #             cv2.imshow('image', imgView)
    #             if self.video_writer is not None:
    #                 cv2.putText(left_image_view, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 cv2.putText(right_image_view, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #                 self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
    #             cv2.waitKey(5)
    #             break
    #         cv2.imshow('image', imgView)
    #         if self.video_writer is not None:
    #             self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
    #             cv2.imshow('Stereo', self.video_operations.image_concat(left_image_view, right_image_view))
    #         key = cv2.waitKey(5)
    #         if key == ord('q'):
    #             break
    #     cv2.destroyAllWindows()

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
        
        segmented_labels = np.array(GMM_Labels).reshape(Shape[0], Shape[1])
        
        self.__get_most_valued_gmm_labels(segmented_labels)
        segmented = self.__get_two_comp_segmentation(segmented_labels)

        if (Save_model):
            # save the model to disk
            filename = 'hand_gmm_model.sav'
            pickle.dump(self.GMM_Model, open(filename, 'wb'))
            filename = 'hand_best_labels.sav'
            pickle.dump(self.two_comp_label_list, open(filename, 'wb'))

    
        fig = plt.figure()
        ax = fig.add_subplot(2, 3, 1)
        imgplot = plt.imshow(segmented_labels+1)
        ax.set_title(f"Segmented with {n_components} components")
        ax = fig.add_subplot(2, 3, 2)
        imgplot = plt.imshow(cv2.cvtColor(GMM_image, cv2.COLOR_BGR2RGB))
        ax.set_title('Original')
        ax = fig.add_subplot(2, 3, 3)
        imgplot = plt.imshow(segmented)
        ax.set_title(f"Segmented with 2 components")
        ax = fig.add_subplot(2, 3, 4)
        imgplot = plt.imshow(self.crop_by_mask_segmented_labels)
        ax.set_title(f"Hand crop segmention")
        ax = fig.add_subplot(2, 3, 5)
        imgplot = plt.imshow(self.invert_crop_by_mask_segmented_labels)
        ax.set_title(f"Invert hand crop segmention")
        plt.show()
        
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
        print(label_list)
        return label_list
    
    def __get_most_valued_gmm_labels(self, n_comp_segmented_img: np.array):
        full_mask = np.zeros(n_comp_segmented_img.shape).astype(np.uint8)
        full_mask[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = self.__get_hand_mask()
        ret, mask = cv2.threshold(full_mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img = n_comp_segmented_img + 1
        self.crop_by_mask_segmented_labels = cv2.bitwise_and(img, img, mask = mask)
        self.invert_crop_by_mask_segmented_labels = cv2.bitwise_and(img, img, mask = mask_inv)
        good_label_list = self.__count_labels(self.crop_by_mask_segmented_labels)
        bad_label_list = self.__count_labels(self.invert_crop_by_mask_segmented_labels)
        self.two_comp_label_list =  self.__remove_bad_labels(good_label_list, bad_label_list)
        print(self.two_comp_label_list)
    
    def __get_hand_mask(self, prev: bool = False):
        img_fill_holes = ndimage.binary_fill_holes(255 - self.mask).astype(np.uint8)
        norm_image = cv2.normalize(img_fill_holes, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        if (prev):
            cv2.imshow("filled", norm_image)
        return norm_image
    
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
    
    def __get_two_comp_segmentation(self, n_comp_segmented_img: np.array):
        reduced_GMM_Labels_segmented_img = np.zeros(n_comp_segmented_img.shape)
        reduced_GMM_Labels_segmented_img[np.isin(n_comp_segmented_img, self.two_comp_label_list)] = 1
        return reduced_GMM_Labels_segmented_img
    


if __name__ == "__main__":
    
    video = Video_operations()
    cal = Calibration(video)
    
    gstreamer_writer = video.open_gstreamer_video_writer("192.168.0.144")
    gstreamer_capture = video.open_gstreamer_video_capture(flip=True)
    video.start_thread_record_view_send(cal.GMM_calibrate)
    # video.view_and_send_video(gstreamer_capture, gstreamer_writer, test_function)
    video.close_gstreamer_video_writer()
    video.close_gstreamer_video_capture()
    
    

