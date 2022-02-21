from aux_functions import *


def hands_matches_points(mask_l, mask_r, im_l, im_r, threshold=10, num_matches=35, nfeatures=6000, show_matches=False, show_kp=False):
    """ # parameters for the feature detection
        self.threshold = 10  # 0, 5, 10, 20
        self.num_matches = 35  # at least 4
        self.nfeatures = 6000  # try as less nfeatures as possible """  # TODO doc good
    # detect features
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=False)
    orb = cv2.ORB_create(nfeatures=nfeatures, nlevels=0, scaleFactor=0)  # no need for different levels and scale

    kp1 = fast.detect(im_l, mask_l)  # detect only points that are in the mask
    kp2 = fast.detect(im_r, mask_r)  # detect only points that are in the mask

    if show_kp:
        img2 = im_l.copy()
        for marker in kp1:
            img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        cv2.imshow("Detected FAST keypoints on the mask", img2)

    if len(kp1) == 0 or len(kp2) == 0:
        # print("no good key points")
        return np.zeros((0, 2)), np.zeros((0, 2))

    kp1, des1 = orb.compute(im_l, kp1)
    kp2, des2 = orb.compute(im_r, kp2)
    if (des1 is None or des2 is None):
        return np.array([]), np.array([])

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:num_matches]
    if show_matches:
        img3 = cv2.drawMatches(im_l, kp1, im_r, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Selected keypoints", img3)
    left_matches = np.array([kp1[match.queryIdx].pt for match in matches])  # left img matched points
    right_matches = np.array([kp2[match.trainIdx].pt for match in matches])  # right img matched points

    # left_matches_y = np.array(sorted(left_matches, key=lambda x: x[1]))
    # left_matches_x = np.array(sorted(left_matches, key=lambda x: x[0]))
    # right_matches_y = np.array(sorted(right_matches, key=lambda x: x[1]))
    # right_matches_x = np.array(sorted(right_matches, key=lambda x: x[0]))
    # left_matches = np.concatenate((left_matches_x[:4] ,left_matches_x[num_matches-4:],left_matches_y[:4],left_matches_y[num_matches-4:]), axis=0)
    # right_matches = np.concatenate((right_matches_x[:4] ,right_matches_x[num_matches-4:],right_matches_y[:4],right_matches_y[num_matches-4:]), axis=0)

    return left_matches, right_matches


def hand_dist_map(src_points, dst_points, mask_src, mask_dst, correlation_tolerance=0.8, show=False):
    dist_map = np.zeros(mask_src.shape[0:2])
    if src_points.shape[0] < 4:  # if there are no matches (may be no hands detected)
        return dist_map
    # transform points - we need to transform all the points in the hand segmentation map
    tform_mat, status = cv2.findHomography(src_points, dst_points)

    temp = np.where(mask_src)
    src = np.transpose(np.array([temp[1], temp[0]]))
    dst = projective_points_tform(tform_mat, src)

    # measure how good was the transformation and show the transformed seg map
    jaccard = measure_transformation(mask_src, mask_dst, tform_mat, name="hand1", show=show)

    if jaccard < correlation_tolerance:
        return dist_map  # map of zeros


    #     DISTANCES
    distances = calc_distance(src[:, 0], dst[:, 0], mask_src.shape[0])
    dist_map[src[:, 1], src[:, 0]] = distances
    return dist_map

