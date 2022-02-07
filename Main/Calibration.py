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
    def __init__(self, video_capture, video_writer = None, stereo: bool = False):
        self.flag = 0
        self.GMM_Image = np.zeros((400, 400))
        self.GMM_Model = None
        self.components_number = 4
        
        self.video_operations = Video_operations()
        self.stereo = stereo
        self.video_capture = video_capture
        self.video_writer = video_writer
        
        cap = self.video_capture
        while True:
            suc, frame = cap.read()
            if suc is True:
                break
        if stereo:
            zero_image = np.zeros(self.video_operations.get_left_image(frame).shape)
        else:
            zero_image = np.zeros(frame.shape)
        self.roi = [zero_image.shape[0] - 340, zero_image.shape[0] - 120, zero_image.shape[1] - 240, zero_image.shape[1] - 20]  # [y_start, y_end, x_start, x_end]
        self.add_values_to_roi = [-40, 60, -100, 0]
        cropPrev = zero_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        self.crop_shape = cropPrev.shape
        handExample = cv2.imread('handExample.jpeg')
        handExample = cv2.resize(handExample, (self.crop_shape[1], self.crop_shape[0]))
        handExampleGray = cv2.cvtColor(handExample, cv2.COLOR_BGR2GRAY)
        ret, self.mask = cv2.threshold(handExampleGray, 10, 255, cv2.THRESH_BINARY)

    def timer_sec(self):
        time.sleep(1)
        self.flag += 1
        time.sleep(1)
        self.flag += 1
        time.sleep(1)
        self.flag += 1
        time.sleep(1)
        self.flag += 1

    def capture_hand(self, print_roi_match: bool = False, flip_camera: bool = False):
        
        cap = self.video_capture
        while True:
            suc, prev = cap.read()
            if suc is True:
                break
        if (self.stereo):
            prev = prev[:, 0:int(prev.shape[1] / 2), :]  # left image
        if (flip_camera):
            prev = cv2.flip(prev, 1)
        
        cropPrev = prev[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        prevGray = cv2.cvtColor(cropPrev, cv2.COLOR_BGR2GRAY)
        numPixels = cropPrev.shape[0] * cropPrev.shape[1]
        tol = numPixels / 10
        change = 0
        startTimer = 0
        t1 = threading.Thread(target=self.timer_sec)
        while cap.isOpened():
            suc, img = cap.read()
            if (self.stereo):
                left_image_view = self.video_operations.get_left_image(img)
                right_image_view = self.video_operations.get_right_image(img)
                img = img[:, 0:int(img.shape[1] / 2), :]  # left image
            if (flip_camera):
                img = cv2.flip(img, 1)
            imgView = np.array(img, copy=True)
            # cv2.rectangle(imgView, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)
            # cropImg_bigger = img[self.roi[0] + self.add_values_to_roi[0]:self.roi[1] + self.add_values_to_roi[1], self.roi[2] + self.add_values_to_roi[2]:self.roi[3]]
            # cropImg_bigger = img

            cropImg = img[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            cropImgView = imgView[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            cropImgView = cv2.bitwise_and(cropImgView, cropImgView, mask=self.mask)
            imgView[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = cropImgView
            if (self.stereo):
                cropImg_right_View = right_image_view[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                cropImg_left_View = left_image_view[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                cropImg_left_View = cv2.bitwise_and(cropImg_left_View, cropImg_left_View, mask=self.mask)
                cropImg_right_View = cv2.bitwise_and(cropImg_right_View, cropImg_right_View, mask=self.mask)
                left_image_view[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = cropImg_left_View
                right_image_view[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]] = cropImg_right_View

            imgGray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            subImg = cv2.subtract(imgGray, prevGray)
            y = subImg.reshape(1, -1)
            change = (y >= 10).sum()
            if(print_roi_match):
                print(tol)
                print(change)
            if change >= tol:
                startTimer += 1
            if startTimer == 1:
                t1.start()
            if self.flag == 1:
                cv2.putText(imgView, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', imgView)
                if self.video_writer is not None:
                    cv2.putText(left_image_view, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(right_image_view, '3', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))

            if self.flag == 2:
                cv2.putText(imgView, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', imgView)
                if self.video_writer is not None:
                    cv2.putText(left_image_view, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(right_image_view, '2', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
            if self.flag == 3:
                cv2.putText(imgView, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('image', imgView)
                if self.video_writer is not None:
                    cv2.putText(left_image_view, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(right_image_view, '1', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
            if self.flag == 4:
                cv2.putText(imgView, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.GMM_Image = img
                cv2.imshow('image', imgView)
                if self.video_writer is not None:
                    cv2.putText(left_image_view, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(right_image_view, 'Image saved', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
                cv2.waitKey(5)
                break
            cv2.imshow('image', imgView)
            if self.video_writer is not None:
                self.video_writer.write(self.video_operations.image_concat(left_image_view, right_image_view))
                cv2.imshow('Stereo', self.video_operations.image_concat(left_image_view, right_image_view))
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

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

    def load_image_and_prepare_for_segmentation(self, path: str):
        image = cv2.imread(path)
        dim = (640, 480)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        roi = [140, 360, 400, 620]  # [y_start, y_end, x_start, x_end]
        cropImg_bigger = resized[roi[0] + self.add_values_to_roi[0]:roi[1] + self.add_values_to_roi[1], roi[2] + self.add_values_to_roi[2]:roi[3]]
        return cropImg_bigger
    


if __name__ == "__main__":
    
    video_operations = Video_operations()
    
    # video_cap = cv2.VideoCapture(0)
    video_cap = video_operations.open_gstreamer_video_capture()
    video_write = video_operations.open_gstreamer_video_writer()
    cal = Calibration(video_cap, stereo=True, video_writer=video_write)
    # cal = Calibration(video_cap)

    
    cal.capture_hand(print_roi_match=True, flip_camera=True)
    # cal.capture_hand(cal.video_operations.open_gstreamer_video_capture(), cal.video_operations.open_gstreamer_video_writer(), stereo=True, print_roi_match=False, flip_camera=False)

    cal.gmm_train(cal.GMM_Image, Save_model=False)
    # cal.video_operations.close_gstreamer_video_capture()
    # cal.video_operations.close_gstreamer_video_writer()
    video_cap.release()
    video_write.release()
    # img = cal.load_image_and_prepare_for_segmentation(os.path.join("Video_and_picture_examples","test.png"))
    # cal.gmm_train(img, Save_model=False)
    # cal.load_saved_model('hand_gmm_model.sav')
    
    

