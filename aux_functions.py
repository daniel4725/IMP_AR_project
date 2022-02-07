import cv2
import numpy as np
from skimage.measure import label, regionprops
import math
from scipy.spatial import ConvexHull
import time
import matplotlib.pyplot as plt


def projective_points_tform(tform_mat, points, inverse=False):
    if inverse:
        tform_mat = np.linalg.inv(tform_mat)
    homogeneous = np.c_[points, np.ones(len(points))]     # create homogeneous coordinates
    tformed_homogeneous = np.dot(tform_mat, homogeneous.T).T  # multiply by the matrix
    tformed = (tformed_homogeneous / tformed_homogeneous[:, 2, None])[:, :2]  # divide by z coordinate and take x and y only
    return np.round(tformed).astype('int')


def drop_outbound_points(shape, points):
    """ points are [x,y] and shape is the shape ot the matrix """
    points = points[points[:, 0] >= 0]
    points = points[points[:, 1] >= 0]
    points = points[points[:, 0] < shape[1]]
    points = points[points[:, 1] < shape[0]]
    return points


def ConvexHullFill(segmentation):
    """
    fills the convex hull with white
    :param segmentation: segmentation map that has one component (2D 1 channel)
    :return: the filled convex hull of the segmentation
    """
    temp = np.where(segmentation == segmentation.max())  # all of the points that are TRUE
    points = np.transpose(np.array([temp[1], temp[0]]))
    hull = ConvexHull(points)  # finds the convex hull
    polygon = list(points[hull.vertices])
    center = (sum([p[0] for p in polygon]) / len(polygon), sum([p[1] for p in polygon]) / len(polygon))
    polygon.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))  # sort by polar angle
    contours = np.array(polygon)
    filled_img = np.zeros(segmentation.shape)
    cv2.fillPoly(filled_img, pts=[contours], color=(255,255,255))  # fills the hull
    return filled_img


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC  TODO use try maybe???
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def getCC_labels_by_size(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC   TODO use try may be???
    bincount_lst = list(enumerate(np.bincount(labels.flat)))[1:]  # list of the count
    bincount_lst = np.array(sorted(bincount_lst, key=lambda item: -item[1]))  # sort by the number of pixels per each label
    labels_by_size = bincount_lst[:, 0]
    bincount_by_size = bincount_lst[:, 1]
    return labels_by_size, bincount_by_size, labels


def calc_distance(x_left, x_right, width):
    mid_left = width//2
    mid_right = width//2
    # ------- camera parameters----------
    alpha = 90  # Camera field of view in the horizontal plane [degrees]
    f_pixel = (width*0.5)/np.tan(alpha*0.5*np.pi/180)  # focal length in pixels
    base = 12
    # ------------disparity---------
    xL = x_left - mid_left
    xR = x_right - mid_right
    disparity = xL - xR
    #  -------------------------
    z = f_pixel * base/(disparity + 10**-13)  # to avoid division by 0
    if z.all() <= 0:
        z = 0
    return z


def my_jaccard_score(seg_map1, seg_map2):
    a = (seg_map1 != 0)
    b = (seg_map2 != 0)
    intersection = (a*b).sum()
    union = (a+b).sum()
    return intersection/union


def measure_transformation(mask_src, mask_dest, tform_mat, name="", show=False):
    tformed_src = cv2.warpPerspective(mask_src, tform_mat, (mask_dest.shape[1], mask_dest.shape[0]))
    jaccard = my_jaccard_score(tformed_src, mask_dest)

    if show:
        cv2.imshow(f'{name}: source', mask_src)
        cv2.imshow(f'{name}: tformed_src', tformed_src)
        cv2.imshow(f'{name}: destenation', mask_dest)
        # print(f"{name}: how many tformed_src landed on dest: {src_on_dst:.2f}%")
        # print(f"{name}: how much dest area tformed_src covered: {dst_area_covered:.2f}%")
        print(f"{name}: jaccard: {jaccard:.2f}")
    return jaccard


def touching_indexes(table_d_map, hands_d_map, tolerance=2, show=False):
    """
    extracts the touching indexes according to the table and the hands distances map
    :param table_d_map: table distances map
    :param hands_d_map: hands distances map
    :param tolerance: the distance (approximately and in cm) that is considered a touch
    :param show: if True, plots the touches mask
    :return: the centroid of each touch
    """
    touch_idxs = np.zeros((0, 2), dtype='int')
    distances = abs(hands_d_map - table_d_map)
    touch_mask = (distances < tolerance) * (hands_d_map > 0)
    if show:
        cv2.imshow("touch_mask", touch_mask.astype('uint8') * 255)
    label_img = label(touch_mask)  # label all the different touches
    regions = regionprops(label_img)  # properties of all touches
    for props in regions:
        y, x = props.centroid
        touch_idxs = np.append(touch_idxs, np.array([[x, y]], dtype='int'), axis=0)
    return touch_idxs


class ImageDistorter:
    def __init__(self, src):
        width = src.shape[1]
        height = src.shape[0]
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

    def distort_and_concat(self, im_l, im_r):
        dst_l = cv2.undistort(im_l, self.cam, self.distCoeff)
        dst_r = cv2.undistort(im_r, self.cam, self.distCoeff)
        return np.concatenate([dst_l, dst_r], axis=1)


if __name__ == "__main__":
    print("Hi, im aux_functions")

