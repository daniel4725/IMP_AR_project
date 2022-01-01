import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.measure import label
import math
from scipy.spatial import ConvexHull
from itertools import combinations, product


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0)  # assume at least 1 CC
    bincount_lst = list(enumerate(np.bincount(labels.flat)))[1:]  # list of the count
    bincount_lst = sorted(bincount_lst, key=lambda item: -item[1])  # sort by the number op pixels per each label
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC * segmentation
def ConvexHullFill(segmentation):
    """
    fills the convex hull with white
    :param segmentation: segmentation map that has one component
    :return: the filled convex hull of the segmentation
    """
    temp = np.where(segmentation == segmentation.max()) # all of the points that are TRUE
    points = np.transpose(np.array([temp[1], temp[0]]))
    hull = ConvexHull(points)  # finds the convex hull
    polygon = list(points[hull.vertices])
    center = (sum([p[0] for p in polygon]) / len(polygon), sum([p[1] for p in polygon]) / len(polygon))
    polygon.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))  # sort by polar angle
    contours = np.array(polygon)
    filled_img = np.zeros((segmentation.shape[0], segmentation.shape[1]))
    cv2.fillPoly(filled_img, pts=[contours], color=(255, 255, 255))  # fills the hull
    return filled_img
for i in range(2,10):
    name= f'table{i}.jpeg'
    image = cv2.imread(name)
    image = cv2.resize(image, (960, 960))                # Resize image

    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    R, G, B = cv2.split(image)
    height, width =G.shape
    #---for G channel----
    blur_G = cv2.medianBlur(G, 7)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen_G = cv2.filter2D(blur_G, -1, sharpen_kernel)
    #---for B channel----
    blur_B = cv2.medianBlur(B, 7)
    sharpen_B = cv2.filter2D(blur_B, -1, sharpen_kernel)
    #---for R channel----
    blur_R = cv2.medianBlur(R, 7)
    sharpen_R = cv2.filter2D(blur_R, -1, sharpen_kernel)

    #-------thresh--------
    th1, threshG = cv2.threshold(sharpen_G, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2, threshB = cv2.threshold(sharpen_B, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3, threshR = cv2.threshold(sharpen_R, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshG = (threshG/255)
    threshB = (threshB/255)
    threshR = (threshR/255)
    thresh = cv2.bitwise_and(threshG, threshB)
    thresh = cv2.bitwise_and(thresh, threshR)

    #----------morphology---------
    line_min_width = 15
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal_v)
    img_bin_final = cv2.bitwise_and(img_bin_h, img_bin_v)

    #-------close morph--------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    close = cv2.morphologyEx(img_bin_final, cv2.MORPH_CLOSE, kernel, iterations=2)

    #----emphasis  edges------
    edges = cv2.Canny(image, 40, 50, apertureSize=3)
    kernel = np.ones((5,5),np.uint8)
    dilated_value = cv2.dilate(edges,kernel,iterations = 1)
    dilated_value = 1-(dilated_value/255)
    edges_thr = cv2.bitwise_and(dilated_value, close)

    #find largest compponent
    final = getLargestCC(edges_thr)

    final_fill = ConvexHullFill(final)

    #--------show images--------
    # cv2.imshow('final', final)
    # cv2.imshow('edges', edges_thr)
    # cv2.imshow('final_fill', final_fill)
    # cv2.waitKey()
    cv2.imwrite("test.jpg", final_fill)
    test = cv2.imread("test.jpg")

    #------contours and canny--------
    edges_seg = cv2.Canny(test, 40, 50, apertureSize=3)
    contours, hierarchy = cv2.findContours(edges_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines_img = np.zeros(image.shape).astype('uint8')
    cv2.drawContours(lines_img, contours, -1, (255,255,255), 1)
    #cv2.imshow('lines_img', lines_img)
    # cv2.imwrite("lines_img.jpg", lines_img)

    #-------finding corners-----(dani)
    def line(p1, p2):
        """ line =  t(p1-p2) + p1 = t*direction + offset for any t"""
        direction = p1 - p2
        offset = p1
        return direction, offset

    def intersection(L1, L2):
        """ solve L1 = L2 => t*direction1 + offset1 = s*direction2 + offset2 """
        a = np.transpose(np.array([L1[0], -L2[0]]))
        b = L2[1] - L1[1]
        try:
            t, s = np.linalg.solve(a, b)
            intersect = t * L1[0] + L1[1]
            return intersect.astype(int)
        except:  # there is no intersection
            return np.array(False)

    def in_image_range(point, img_shape):
        if point[0] < img_shape[1] and point[0] >= 0 and point[1] < img_shape[0] and point[1] >= 0:
            return True
        else:
            return False

    def get_table_corners(table_segmentation):
        # TODO add inputs, and optimize thresholds and function parameters
        num_of_lines = 4
        hough_rho = 1
        hough_theta = (np.pi / 180) * 1
        hough_thresh = 50

        # TODO maby change the points?? + 1000??

        edges = table_segmentation
        hlines = cv2.HoughLines(image=edges[:, :, 0], rho=hough_rho, theta=hough_theta, threshold=hough_thresh)
        lines = []
        for i in range(num_of_lines):
            for rho, theta in hlines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lines.append(line(np.array([x1, y1]), np.array([x2, y2])))
                #cv.line(edges, (x1, y1), (x2, y2), (0, 0, 35 + 35 * i), 3)  # TODO only for tests
        #cv.imshow("ss", edges)

        for fourlines in combinations(lines, 4):
            intersect = []
            for L1, L2 in combinations(fourlines, 2):
                point = intersection(L1, L2)
                if point.all() and in_image_range(point, table_segmentation.shape[0:2]):
                    intersect.append(point)
            # TODO if there are no 4 intersection

        center = (sum([p[0] for p in intersect]) / len(intersect), sum([p[1] for p in intersect]) / len(intersect))
        intersect.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))  # sort by polar angle

        return np.array(intersect)

    #---------get table corners

    table_corners = get_table_corners(lines_img)
    plt.figure()
    plt.plot(table_corners[:,0], table_corners[:,1], marker='v', color="white")
    plt.imshow(image)
    plt.show()
