from user_interface import Application
from aux_functions import *
from table_handling import TableMap, CornersFollower, get_table_corners
from hands_handling import hands_matches_points, hand_dist_map
import threading
from StateMachine import StateMachine
from HandOperations import HandOperations
import os
import pickle

# TODO:
#  1. handle the sleeve mask
#  2. renew seg mask - differet function, delay there and when to do?.

def write(im_l, im_r, txt="3D! Hello"):  # TODO delete
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = 80
    offset = 200
    # org
    org_r = (start, 100)
    org_l = (start + offset, 100)
    # fontScale
    fontScale = 1
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    im_l = cv2.putText(im_l, txt, org_l, font, fontScale, color, thickness, cv2.LINE_AA, False)
    im_r = cv2.putText(im_r, txt, org_r, font, fontScale, color, thickness, cv2.LINE_AA, False)
    return im_l, im_r

class AR:
    def __init__(self, im_l, im_r):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_output")
        with open(os.path.join(directory, 'corner_follower_l.sav'), 'rb') as handle:
            self.corner_follower_l = pickle.load(handle)
        with open(os.path.join(directory, 'corner_follower_r.sav'), 'rb') as handle:
            self.corner_follower_r = pickle.load(handle)

        # self.corner_follower_l = CornersFollower(im_l, show=True,
        #                                     name="c_follow_l")  # Create the corner follower for left image
        # self.corner_follower_r = CornersFollower(im_r, show=True,
        #                                     name="c_follow_r")  # Create the corner follower for right image

        # Objects
        self.renew_seg_countdown_clk = CountDown()
        self.hand_operations = HandOperations()
        self.table_map = TableMap(self.corner_follower_l.current_corners, self.corner_follower_r.current_corners,
                             shape=im_l.shape)
        # get the table's distances map
        self.dist_map_table = self.table_map.table_dist_map(self.corner_follower_l.current_corners,
                                                  self.corner_follower_r.current_corners, show=False, check_err=False)
        self.dist_map_hands = None

        self.application = Application(self.table_map.map)
        self.state_machine = StateMachine(self.hand_operations, im_l.shape)

        # corner following fields:
        self.new_corners_r, self.changed_r = np.zeros((4, 2)), False
        self.renew_table_segmentation = False
        self.renewing_t_seg = False
        self.renew_seg_counter = 1
        self.re_corners_r, re_seg_img_r = 0, 0

        # TODO get from where??
        self.im_l = None
        self.im_r = None
        self.mask_l = None
        self.mask_r = None

        # TODO DELETE AND CHANGE
        self.mr = 9  # TODO delete!! also from func

    def get_hand_r_seg(self):
        self.mr = self.hand_operations.get_hand_mask(self.im_r)
        # TODO change to mask_r

    def renew_seg_right(self, scale):
        self.re_corners_r, self.re_seg_img_r = get_table_corners(self.im_r, scale=scale)  # TODO scale?

    def hands_distance_calc(self):
        # find matches hands points
        src_points, dst_points = hands_matches_points(self.mask_l, self.mask_r, self.im_l, self.im_r)

        # get the hand's distances map
        self.dist_map_hands = hand_dist_map(src_points, dst_points, self.mask_l, self.mask_r,
                                       correlation_tolerance=0.4, show=False)

    def follow_right_corners(self):
        self.new_corners_r, self.changed_r = self.corner_follower_r.follow(self.im_r, self.mask_r, show_out=False)

    def table_distance_calc(self):
        # following right and left corners in parallel:
        former_corners = (self.corner_follower_l.current_corners.copy(), self.corner_follower_r.current_corners.copy())
        right_corners_thread = threading.Thread(target=self.follow_right_corners)
        right_corners_thread.start()
        new_corners_l, changed_l = self.corner_follower_l.follow(self.im_l, self.mask_l, show_out=False)
        right_corners_thread.join()

        if (changed_l and not self.changed_r) or (self.changed_r and not changed_l):
            # tracking error - change the corners back
            self.corner_follower_l.current_corners = former_corners[0]
            self.corner_follower_r.current_corners = former_corners[1]
        elif changed_l and self.changed_r:
            self.dist_map_table, valid = self.table_map.table_dist_map(new_corners_l, self.new_corners_r, check_err=True,
                                                             former_dist_map=self.dist_map_table, show=False)
            if not valid:
                # tracking error - renewed table segmentation is needed
                self.renew_table_segmentation = True

    def get_AR(self, im_l, im_r, mask_l=None, mask_r=None):
        self.im_l = im_l
        self.im_r = im_r
        self.mask_l = mask_l
        self.mask_r = mask_r
        # --------- bad table following for some time ----
        if self.renew_table_segmentation or self.renewing_t_seg:
            if self.renew_table_segmentation:
                self.renewing_t_seg = True
                self.renew_table_segmentation = False
                self.renew_seg_countdown_clk.start_countdown(self.renew_seg_counter)

            if self.renew_seg_countdown_clk.check() == 0:
                scale = 2  # TODO scale?
                renew_seg_r = threading.Thread(target=self.renew_seg_right, args=(scale,))
                renew_seg_r.start()
                re_corners_l, re_seg_img_l = get_table_corners(im_l, scale=scale, name="renew_l", show=True)
                renew_seg_r.join()
                self.corner_follower_l.current_corners, self.table_map.corners_l = re_corners_l, re_corners_l
                self.corner_follower_r.current_corners, self.table_map.corners_r = self.re_corners_r, self.re_corners_r
                self.renewing_t_seg = False
            else:
                factor = 1.2
                dim = (int(im_l.shape[1] * factor * 0.6), int(im_l.shape[0] * factor))
                resized_l = cv2.resize(im_l, dim, interpolation=cv2.INTER_AREA)
                resized_r = cv2.resize(im_r, dim, interpolation=cv2.INTER_AREA)
                txt = f'SG in: {self.renew_seg_countdown_clk.check()}'
                write(resized_l, resized_r, txt=txt)
                out = np.concatenate([resized_l, np.zeros((dim[1], 10, 3), dtype='uint8'), resized_r], axis=1)
                cv2.imshow('out_stream', out)
                cv2.waitKey(70)  # TODO delete line
            return im_l, im_r

        # start a thread that updates the table map using screen shoot
        self.table_map.update_whole_map(application=self.application)

        # get the segmentation maps of the hands
        # TODO ------------------------ here:  -------------------------------
        #  using im_l and im_r extracting mask_l and mask_r
        hand_seg_r_thread = threading.Thread(target=self.get_hand_r_seg)
        hand_seg_r_thread.start()
        ml = self.hand_operations.get_hand_mask(im_l)
        hand_seg_r_thread.join()
        cv2.imshow("makim", np.concatenate([ml, self.mr], axis=1))
        # mask_l, mask_r = hand_operations.get_hand_mask(im_l), hand_operations.get_hand_mask(im_r)

        # ---------------------- state machine and distances map (hands and table) calculations --------------------
        # starts the hands and the table distance map calculations
        hands_thread = threading.Thread(target=self.hands_distance_calc)
        table_thread = threading.Thread(target=self.table_distance_calc)
        table_thread.start()
        hands_thread.start()
        # TODO ------------------------ here:  -------------------------------
        #  calculate the state according to former one and calculate thing that are in the stage
        #  4. change state accordingly, update the the relevant app (in tablemap
        #  and other relevant objects)
        self.state_machine.state_operations(self.im_l, self.mask_l, self.application)

        table_thread.join()
        hands_thread.join()

        # TODO ------------------------ here:  -------------------------------
        #  do stuff according to each state

        # TODO 1.
        # ------------------  compare both distances map to find the touching indexes --------------------
        if self.state_machine.state in self.state_machine.ACTIVE_STATES:
            touch_idxs = touching_indexes(self.dist_map_table, self.dist_map_hands, tolerance=5, show=False)  # TODO: tolerance
        else:
            touch_idxs = np.zeros((0, 2))  # no touches

        # TODO ------------------------ here:  -------------------------------
        #  after doing stuff - project the map to the table
        # ---------------- estimating the transform and projecting the map to the real table  ------------------
        im_l, im_r = self.table_map.project_map2real(im_l, im_r, mask_l, mask_r, application=self.application,
                                                real_touch_idxs=touch_idxs, touch_indicator=True, show_touch=False)

        self.state_machine.add_text(im_l)
        return im_l, im_r
