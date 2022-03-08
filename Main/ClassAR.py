from user_interface import Application
from aux_functions import *
from table_handling import TableMap, CornersFollower, get_table_corners
from hands_handling import hands_matches_points, hand_dist_map
import threading
from StateMachine import StateMachine
from HandOperations import HandOperations
import os


class AR:
    def __init__(self, im_l, im_r, crop_x: int = 0, static_cam=False):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_output")
        # with open(os.path.join(directory, 'corner_follower_l.sav'), 'rb') as handle:
        #     self.corner_follower_l = pickle.load(handle)
        # with open(os.path.join(directory, 'corner_follower_r.sav'), 'rb') as handle:
        #     self.corner_follower_r = pickle.load(handle)
        self.use_sleeves = True

        self.corner_follower_l = CornersFollower(im_l, static_cam=static_cam, show=True,
                                            name="c_follow_l")  # Create the corner follower for left image
        self.corner_follower_r = CornersFollower(im_r, static_cam=static_cam, show=True,
                                            name="c_follow_r")  # Create the corner follower for right image

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
        self.state_machine = StateMachine(self.hand_operations, im_l.shape, crop_x=crop_x)

        # corner following fields:
        self.new_corners_r, self.changed_r = np.zeros((4, 2)), False
        self.renew_table_segmentation = False
        self.renewing_t_seg = False
        self.renew_seg_counter = 3
        self.re_corners_r, re_seg_img_r = 0, 0

        self.im_l = None
        self.im_r = None
        self.mask_l = None
        self.mask_r = None
        self.sleeve_l = None
        self.sleeve_r = None
        self.cover_l, self.cover_r = None, None

    def get_hand_r_seg(self):
        self.mask_r = self.hand_operations.get_hand_mask(self.im_r)

    def get_sleeve_r_seg(self):
        self.sleeve_r = self.hand_operations.get_hand_mask(self.im_r, get_sleeve=True)

    def get_sleeve_l_seg(self):
        self.sleeve_l = self.hand_operations.get_hand_mask(self.im_l, get_sleeve=True)

    def renew_seg_right(self, scale):
        self.re_corners_r, self.re_seg_img_r = get_table_corners(self.im_r, scale=scale)

    def hands_distance_calc(self):
        # find matches hands points
        src_points, dst_points = hands_matches_points(self.mask_l, self.mask_r, self.im_l, self.im_r)

        # get the hand's distances map
        self.dist_map_hands = hand_dist_map(src_points, dst_points, self.mask_l, self.mask_r,
                                       correlation_tolerance=0.4, show=False)

    def follow_right_corners(self):
        self.new_corners_r, self.changed_r = self.corner_follower_r.follow(self.im_r, self.cover_r, show_out=False)

    def table_distance_calc(self):
        # following right and left corners in parallel:
        former_corners = (self.corner_follower_l.current_corners.copy(), self.corner_follower_r.current_corners.copy())
        right_corners_thread = threading.Thread(target=self.follow_right_corners)
        right_corners_thread.start()
        new_corners_l, changed_l = self.corner_follower_l.follow(self.im_l, self.cover_l, show_out=False)
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

    def find_table(self):
        if self.renew_table_segmentation:
            self.renewing_t_seg = True
            self.renew_table_segmentation = False
            self.renew_seg_countdown_clk.start_countdown(self.renew_seg_counter)

        if self.renew_seg_countdown_clk.check() == 0:
            scale = 2
            renew_seg_r = threading.Thread(target=self.renew_seg_right, args=(scale,))
            renew_seg_r.start()
            re_corners_l, re_seg_img_l = get_table_corners(self.im_l, scale=scale, name="renew_l", show=True)
            renew_seg_r.join()
            self.corner_follower_l.current_corners, self.table_map.corners_l = re_corners_l, re_corners_l
            self.corner_follower_r.current_corners, self.table_map.corners_r = self.re_corners_r, self.re_corners_r
            self.renewing_t_seg = False
        else:
            current_count = f'{self.renew_seg_countdown_clk.check()}'
            self.state_machine.write_looking4table(self.im_l, self.im_r, current_count)

    def get_AR(self, im_l, im_r, show_dist_map=False, renew_table_seg=True):
        s = time.time()
        self.im_l = im_l
        self.im_r = im_r
        # --------- bad table following for some time ----
        if (self.renew_table_segmentation or self.renewing_t_seg) and renew_table_seg:
            self.find_table()
            return self.im_l, self.im_r

        # start a thread that updates the table map using screen shoot
        self.table_map.update_whole_map(application=self.application)

        # get the segmentation maps of the hands
        hand_seg_r_thread = threading.Thread(target=self.get_hand_r_seg)
        sleeve_seg_l_thread = threading.Thread(target=self.get_sleeve_l_seg)
        sleeve_seg_r_thread = threading.Thread(target=self.get_sleeve_r_seg)
        hand_seg_r_thread.start()
        sleeve_seg_l_thread.start()
        sleeve_seg_r_thread.start()
        self.mask_l = self.hand_operations.get_hand_mask(im_l)
        hand_seg_r_thread.join()
        sleeve_seg_l_thread.join()
        sleeve_seg_r_thread.join()
        cv2.imshow("hands", np.concatenate([self.mask_l, self.mask_r], axis=1))
        cv2.imshow("sleeves", np.concatenate([self.sleeve_l, self.sleeve_r], axis=1))
        if self.use_sleeves:
            self.cover_l, self.cover_r = self.mask_l + self.sleeve_l, self.mask_r + self.sleeve_r
        else:
            self.cover_l, self.cover_r = self.mask_l, self.mask_r

        # ---------------------- state machine and distances map (hands and table) calculations --------------------
        # starts the hands and the table distance map calculations
        hands_thread = threading.Thread(target=self.hands_distance_calc)
        table_thread = threading.Thread(target=self.table_distance_calc)
        table_thread.start()
        hands_thread.start()
        self.renew_table_segmentation = self.state_machine.state_operations(self.im_l, self.mask_l, application=self.application)

        table_thread.join()
        hands_thread.join()

        # ------------------  compare both distances map to find the touching indexes --------------------
        tip_map = self.hand_operations.get_tip_mask(self.mask_l)
        cv2.imshow('tip', tip_map * 255)
        if self.state_machine.state in self.state_machine.ACTIVE_STATES:
            tip_map = self.hand_operations.get_tip_mask(self.mask_l)
            touch_idxs = touching_indexes(self.dist_map_table, self.dist_map_hands * tip_map, tolerance=4, show=False)
        else:
            touch_idxs = np.zeros((0, 2))  # no touches

        # ---------------- estimating the transform and projecting the map to the real table  ------------------



        im_l, im_r = self.table_map.project_map2real(self.im_l, self.im_r, self.cover_l, self.cover_r, application=self.application,
                                                real_touch_idxs=touch_idxs, touch_indicator=True, show_touch=False)

        self.state_machine.add_text(im_l, left=True)
        self.state_machine.add_text(im_r)


        if True:
            # show distances map
            dist_map = self.dist_map_table.copy()
            dist_map[self.dist_map_hands > 0] = self.dist_map_hands[self.dist_map_hands > 0]
            cv2.imshow('distances', dist_map.astype('uint8') * 2)
        # print(time.time() - s)
        return im_l, im_r
