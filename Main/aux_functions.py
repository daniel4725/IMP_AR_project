import cv2
import numpy as np
from skimage.measure import label, regionprops
import math
from scipy.spatial import ConvexHull
from scipy import stats
import time
import matplotlib.pyplot as plt


def get_corners_with_lin_reg(points):
    """
    extracts 4 corners from set of rectangle points using linear regression
    :param points: rectangle points
    :return: 4 corners of the rectangle
    """
    tmp_lines = points
    outliers_thresh = 1
    new_table_corners = np.zeros((4, 2), dtype=np.int)
    lines_m_b = np.zeros((4, 2))

    points_lst = []
    for i, line in enumerate(tmp_lines):
        z_score = np.abs(stats.zscore(line))  # dropping outliers  
        no_outliers = line[(z_score < outliers_thresh).all(axis=1), :]
        # print(f"{len(no_outliers)/line.shape[0]}")
        # if len(no_outliers)/line.shape[0] < 0.40:  # if there are a lot of outliers takes the former corners
        #     print(f"dropped points: no outliers - {len(no_outliers)/line.shape[0]}")
        #     points_lst.append(self.current_corners[i])
        #     # points_lst.append(no_outliers)
        # else:
        points_lst.append(no_outliers)
        try:
            m, b = np.polyfit(no_outliers[:, 0], no_outliers[:, 1], 1)  # fits a line
        except:
            m = b = 0  # line is vertical
        lines_m_b[i, :] = np.array((m, b))

    for i in range(4):
        m1, b1 = lines_m_b[i]
        m2, b2 = lines_m_b[(i + 1) % 4]

        if abs(m1 - m2) < 10 ** -8:
            # https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
            print('\x1b[0;30;41m' + "LinReg: something went wrong... the lines are paralel" + '\x1b[0m')
            return 0

        y = ((m1 * b2 - b1 * m2) / (m1 - m2)).round()
        x = ((b2 - b1) / (m1 - m2)).round()
        new_table_corners[i, :] = np.array([x, y])

    return new_table_corners.astype(np.int)


def get_corners_with_hough(side_edges):
    """
    extracts 4 corners from set of rectangle points using linear hough lines for each edge
    :param points: 4 edges maps
    :return: 4 corners of the rectangle (lines intersection)
    """
    hough_rho = 1
    hough_theta = (np.pi / 180) * 1
    hough_thresh = 50

    lines = np.zeros((4, 2))
    corners = np.zeros((4, 2))
    for i, side_edge in enumerate(side_edges):
        hlines = cv2.HoughLines(image=side_edge, rho=hough_rho, theta=hough_theta, threshold=hough_thresh)
        rho, theta = hlines[0, 0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = (x0 + 100 * (-b))
        y1 = (y0 + 100 * (a))
        x2 = (x0 - 100 * (-b))
        y2 = (y0 - 100 * (a))
        try:
            m, b = np.polyfit([x1, x2], [y1, y2], 1)  # fits a line
        except:
            m = b = 0  # line is vertical
        lines[i, :] = np.array((m, b))

    for i in range(4):
        m1, b1 = lines[i]
        m2, b2 = lines[(i + 1) % 4]

        if abs(m1 - m2) < 10 ** -8:
            # https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal
            print('\x1b[0;30;41m' + "Hough: something went wrong... the lines are paralel" + '\x1b[0m')
            return 0

        y = ((m1 * b2 - b1 * m2) / (m1 - m2)).round()
        x = ((b2 - b1) / (m1 - m2)).round()
        corners[i, :] = np.array([x, y])

    return corners.astype(np.int)


def projective_points_tform(tform_mat, points, inverse=False):
    """ projective transform for points using tha matrix """
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
    """ get the largest connected componnent """
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def calc_distance(x_left, x_right, width):
    """
    calculates the distances between the camera and the object using disparity between
    the location in the left and right image
    :param x_left: pixel location of an object in the left image
    :param x_right: pixel location of the same object in the right image
    :param width: width of the image
    :return:
    """
    mid_left = width//2
    mid_right = width//2
    # ------- camera parameters----------
    # alpha = 80  # Camera field of view in the horizontal plane [degrees]
    # (width*0.5)/np.tan(alpha*0.5*np.pi/180)
    f_pixel = 350  # focal length in pixels
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
    """    measures how good is the transformation using jaccard score    """
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
    :return: the closest index of each touch
    """
    touch_idxs = np.zeros((0, 2), dtype='int')
    distances = abs(hands_d_map - table_d_map)
    touch_mask = (distances < tolerance) * (hands_d_map > 0)
    if show:
        cv2.imshow("touch_mask", touch_mask.astype('uint8') * 255)
    if touch_mask.any():
        largest_touch = getLargestCC(touch_mask)
        temp = distances * largest_touch
        temp[temp == 0] = tolerance
        closest_touch = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
        touch_idxs = np.array([[closest_touch[1], closest_touch[0]]], dtype='int')
    # label_img = label(touch_mask)  # label all the different touches
    # regions = regionprops(label_img)  # properties of all touches
    # for props in regions:
    #     y, x = props.centroid
    #     touch_idxs = np.append(touch_idxs, np.array([[x, y]], dtype='int'), axis=0)
    return touch_idxs


class CountDown:
    """ class that countdown seconds    """
    def __init__(self):
        self.counter = 0
        self.start_time = 0
        self.starts_from = 0

    def start_countdown(self, num_of_sec):
        self.counter = num_of_sec + 1
        self.starts_from = num_of_sec + 1
        self.start_time = time.time()

    def check(self):
        self.counter = int(self.starts_from - (time.time() - self.start_time))
        if self.counter <= 0:
            self.counter = 0
        return self.counter

if __name__ == "__main__":
    print("Hi, im aux_functions")
    countdown_clk = CountDown()
    countdown_clk.start_countdown(8)
    while 1:
        time.sleep(0.2)
        print(f'\rcounting down: {countdown_clk.check()}', end="")
        if countdown_clk.check() == 0:
            break
    print("\rdone")

