import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os


def get_segmentation(image):
    Shape = image.shape
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    L = np.array(imageLAB[:, :, 0]).flatten()
    a = np.array(imageLAB[:, :, 1]).flatten()
    b = np.array(imageLAB[:, :, 2]).flatten()
    data = np.array([a, b]).transpose()

    n_compononts = 2
    GMM_Model = GaussianMixture(n_components=n_compononts)
    GMM_Model.fit(data)
    GMM_Labels = GMM_Model.predict(data)

    if GMM_Labels[0] == 1:
        GMM_Labels = np.array(GMM_Labels, dtype=bool)
        GMM_Labels = np.invert(GMM_Labels)
        GMM_Labels = np.array(GMM_Labels, dtype=int)

    # to show the results
    segmented = np.array(GMM_Labels).reshape(Shape[0], Shape[1])
    return segmented


def get_hand(segmented, image):
    morph = cv2.normalize(segmented, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    kernel = cv2.getStructuringElement(cv2.MARKER_CROSS, (5, 5))
    # kernel = np.ones((5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    segmented = cv2.normalize(morph, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    ret, mask = cv2.threshold(segmented, 10, 255, cv2.THRESH_BINARY)
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


def execute(path: str):
    for folder in os.listdir(path):
        path_of_folder = os.path.join(path, folder)
        for file in os.listdir(path_of_folder):
            image_path = os.path.join(path_of_folder, file)
            image = cv2.imread(image_path)
            segmented = get_segmentation(image)
            hand = get_hand(segmented, image)
            cv2.imwrite(image_path, hand)


if __name__ == "__main__":
    execute(path='Train')
    # image = cv2.imread('Train/circle/circle_0.jpeg')
    # segmented = get_segmentation(image)
    # hand = get_hand(segmented, image)
    # fig = plt.figure()
    # plt.imshow(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB))
    # plt.show()



