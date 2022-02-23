import os

import cv2
import numpy as np
from aux_functions import CountDown

class StateMachine:
    MAIN_MENU = 0
    PAINT_MENU = 1
    TABLET = 2
    INSERT_SHAPE = 3
    DELETE_SHAPE = 4
    SHAPE_MANIPULATION = 5
    CHANGE_COLOR = 6
    DRAW = 7
    ACTIVE_STATES = [DRAW, SHAPE_MANIPULATION, TABLET, DELETE_SHAPE]

    BACK = -1
    TRIANGLE = 1
    CIRCLE = 2
    SQUARE = 3
    SMILIE = 4

    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4
    PURPLE = 5

    FONT_COLOR = (246, 158, 70)
    COLOR2 = (117, 4, 129)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    size = 0.7
    thick = 1

    LOC_TITLE = np.array([70, 20])
    LOC1 = np.array([-50, 20 + 60])
    LOC2 = np.array([-50, 40 + 60])
    LOC3 = np.array([-50, 60 + 60])
    LOC4 = np.array([-50, 80 + 60])
    LOC5 = np.array([-50, 100 + 60])
    LOC_CENTER_SHAPE = np.array([0, 200])
    LOC_CENTER_SHAPE_MANI = np.array([100, 200])
    LOC_CENTER_SHAPE_INSERT = np.array([150, 200])

    offset = 50

    def __init__(self, hand_operations, shape, crop_x):
        self.state = self.MAIN_MENU
        self.former_state = self.MAIN_MENU
        self.classifier_roi = [130, 230, shape[1]//2 - 60 + 30, shape[1]//2 + 60 + 30]  # [y_start, y_end, x_start, x_end]
        # back_roi = [0, 40, shape[1]-crop_x - 40, shape[1]-crop_x]
        back_roi = [0, 40, crop_x - 40, crop_x + 40 - 40]
        back_dim = (back_roi[3] - back_roi[2], back_roi[1] - back_roi[0])
        self.back_button_roi = back_roi
        renew_table_roi = [0, 40, crop_x + 50, crop_x + 40 + 50]
        self.renew_table_roi = renew_table_roi
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        back_img = cv2.imread(os.path.join(directory, "back.png"))
        self.back_img = cv2.resize(back_img, back_dim, interpolation=cv2.INTER_AREA)
        find_table_img = cv2.imread(os.path.join(directory, "find_table.png"))
        self.find_table_img = cv2.resize(find_table_img, back_dim, interpolation=cv2.INTER_AREA)
        self.in_delay = False
        self.delay_between_states = 2
        self.state_changed = False
        self.saved_choice = 0

        self.img_roi = [0 + 80, 40 + 80, crop_x, crop_x + 40]
        img_dim = (self.img_roi[3] - self.img_roi[2], self.img_roi[1] - self.img_roi[0])
        square_img = cv2.imread(os.path.join(directory, "square.jpeg"))
        self.square_img = cv2.resize(square_img, img_dim, interpolation=cv2.INTER_AREA)
        triangle_img = cv2.imread(os.path.join(directory, "triangle.jpeg"))
        self.triangle_img = cv2.resize(triangle_img, img_dim, interpolation=cv2.INTER_AREA)
        circle_img = cv2.imread(os.path.join(directory, "circle.jpeg"))
        self.circle_img = cv2.resize(circle_img, img_dim, interpolation=cv2.INTER_AREA)
        smilie_img = cv2.imread(os.path.join(directory, "smilie.jpeg"))
        self.smilie_img = cv2.resize(smilie_img, img_dim, interpolation=cv2.INTER_AREA)

        classifier_roi_2_counter = np.array(self.classifier_roi, copy=True)
        classifier_roi_2_counter[3] += self.offset
        self.count_fingers = ClassifierCounter(classifier_roi_2_counter, classifier=hand_operations.get_finger_num)
        self.shape_classifier = ClassifierCounter(classifier_roi_2_counter, classifier=hand_operations.predict_shape)
        self.back_button = Button(self.back_button_roi, "back")
        self.renew_button = Button(self.renew_table_roi, "renew")
        self.countdown_clk = CountDown()

        self.shape_name_dict = {self.TRIANGLE: "triangle", self.CIRCLE: "circle",
                                self.SQUARE: "square", self.SMILIE: "smilie"}

        self.color_name_dict = {self.BLUE: "blue", self.GREEN: "green", self.RED: "red",
                                self.YELLOW: "yellow", self.PURPLE: "Yossi"}

        self.LOC_TITLE[0] += crop_x
        self.LOC1[0] += crop_x
        self.LOC2[0] += crop_x
        self.LOC3[0] += crop_x
        self.LOC4[0] += crop_x
        self.LOC5[0] += crop_x
        self.LOC_CENTER_SHAPE[0] += crop_x
        self.LOC_CENTER_SHAPE_MANI[0] += crop_x
        self.LOC_CENTER_SHAPE_INSERT[0] += crop_x

        self.loc_TITLE = np.array(self.LOC_TITLE, copy=True)
        self.loc1 = np.array(self.LOC1, copy=True)
        self.loc2 = np.array(self.LOC2, copy=True)
        self.loc3 = np.array(self.LOC3, copy=True)
        self.loc4 = np.array(self.LOC4, copy=True)
        self.loc5 = np.array(self.LOC5, copy=True)
        self.loc_CENTER_SHAPE = np.array(self.LOC_CENTER_SHAPE, copy=True)
        self.loc_CENTER_SHAPE_MANI = np.array(self.LOC_CENTER_SHAPE_MANI, copy=True)
        self.loc_CENTER_SHAPE_INSERT = np.array(self.LOC_CENTER_SHAPE_INSERT, copy=True)

    def transition(self, flag, application=None):
        tmp_state = self.state
        if self.state == self.MAIN_MENU:
            if flag == 1:
                self.state = self.PAINT_MENU
                application.run(application.PAINT)
            elif flag == 2:
                self.state = self.TABLET
                application.enable_touch = True
                application.run(application.TABLET)

        elif self.state == self.TABLET and flag == self.BACK:
            self.state = self.MAIN_MENU
            application.enable_touch = False
            application.exit()

        elif self.state == self.PAINT_MENU:
            if flag == self.BACK:
                self.state = self.MAIN_MENU
                application.exit()
            elif flag == 1:
                self.state = self.INSERT_SHAPE
            elif flag == 2:
                self.state = self.DELETE_SHAPE
                application.enable_touch = True
                application.change_paint_state(application.DELETE)
            elif flag == 3:
                self.state = self.SHAPE_MANIPULATION
                application.enable_touch = True
            elif flag == 4:
                self.state = self.CHANGE_COLOR
            elif flag == 5:
                self.state = self.DRAW
                application.change_paint_state(application.DRAW)
                application.enable_touch = True

        elif self.state == self.INSERT_SHAPE:
            if flag == self.BACK:
                self.state = self.PAINT_MENU
            elif flag in self.shape_name_dict.keys():
                # TODO insert the shape
                self.saved_choice = self.shape_name_dict[flag]
                self.state = self.SHAPE_MANIPULATION
                application.change_paint_state(application.INSERT_SHAPE, flag)
                application.enable_touch = True

        elif self.state == self.CHANGE_COLOR:
            if flag == self.BACK:
                self.state = self.PAINT_MENU
            elif flag in self.color_name_dict.keys():
                # TODO insert the shape
                self.saved_choice = self.color_name_dict[flag]
                self.state = self.PAINT_MENU
                application.change_paint_state(application.CHANGE_COLOR, flag)

        elif (self.state in [self.DELETE_SHAPE, self.SHAPE_MANIPULATION, self.DRAW]) and flag == self.BACK:
            if self.state == self.DRAW:
                application.change_paint_state(application.EXIT_DRAW)
            elif self.state == self.DELETE_SHAPE:
                application.change_paint_state(application.EXIT_DELETE)
            application.enable_touch = False
            self.state = self.PAINT_MENU

        self.state_changed = self.state != tmp_state

    def write_looking4table(self, im_l, im_r, counter):
        left_loc = self.LOC_CENTER_SHAPE.copy()
        left_loc[0] += self.offset
        cv2.putText(im_l, "  " + counter, tuple(left_loc), self.font, 3, self.FONT_COLOR, 4, cv2.LINE_4)
        cv2.putText(im_r, "  " + counter, tuple(self.LOC_CENTER_SHAPE), self.font, 3, self.FONT_COLOR, 4, cv2.LINE_4)

        left_loc = self.LOC1.copy()
        left_loc[0] += self.offset
        cv2.putText(im_l, "        Looking for Table...", tuple(left_loc), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
        cv2.putText(im_r, "        Looking for Table...", tuple(self.LOC1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)

        left_loc = self.LOC3.copy()
        left_loc[0] += self.offset
        cv2.putText(im_l, "     Please Move You'r Hands", tuple(left_loc), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
        cv2.putText(im_r, "     Please Move You'r Hands", tuple(self.LOC3), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)

    def add_text(self, frame, left=False):
        self.change_use_locations(left)

        if self.in_delay or self.state_changed:
            if (self.state in [self.SHAPE_MANIPULATION, self.PAINT_MENU]) and (self.saved_choice != 0):
                size = 1
                thick = 2
                location = self.loc3
                spaces = "       "
                color = self.COLOR2
                cv2.putText(frame, spaces + self.saved_choice, tuple(location), self.font, size, color, thick, cv2.LINE_4)
            else:
                self.write_state_on_frame(frame)

        elif self.state == self.MAIN_MENU:
            cv2.putText(frame, "Main Menu", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '1. Paint app', tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '2. Tablet', tuple(self.loc2), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi, left=left)
            self.add_renew_button(frame, left)
        elif self.state == self.PAINT_MENU:
            cv2.putText(frame, "Paint Menu", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '1. Insert shape', tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '2. Delete shape', tuple(self.loc2), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '3. Shape Manipulation', tuple(self.loc3), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '4. Change color', tuple(self.loc4), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '5. Draw', tuple(self.loc5), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            # cv2.putText(frame, 'Put fist to go back ', (20, 120), self.font, 1, self.FONT_COLOR, 2, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi, left=left)
            self.add_back_button(frame, left)
            self.add_renew_button(frame, left)
        elif self.state == self.TABLET:
            cv2.putText(frame, "Tablet", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            self.add_back_button(frame, left)
            self.add_renew_button(frame, left)

        elif self.state == self.INSERT_SHAPE:
            self.draw_shape_images(frame, left)
            spaces = "     "
            cv2.putText(frame, "Insert Shape", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, spaces + 'Smilie', tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, spaces + 'Triangle', tuple(self.loc2), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, spaces + 'Circle', tuple(self.loc3), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, spaces + 'Square', tuple(self.loc4), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi, left=left)
            self.add_back_button(frame, left)
            self.add_renew_button(frame, left)

        elif self.state == self.DELETE_SHAPE:
            cv2.putText(frame, "Delete Shapes", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, 'Touch shape to delete', tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            self.add_back_button(frame, left)
            self.add_renew_button(frame, left)

        elif self.state == self.SHAPE_MANIPULATION:
            cv2.putText(frame, "Shape Manipulation", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, 'Touch shape to move', tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            self.add_back_button(frame, left)
            self.add_renew_button(frame, left)

        elif self.state == self.CHANGE_COLOR:
            cv2.putText(frame, "Change Color", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '1. ' + self.color_name_dict[1], tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '2. ' + self.color_name_dict[2], tuple(self.loc2), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '3. ' + self.color_name_dict[3], tuple(self.loc3), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '4. ' + self.color_name_dict[4], tuple(self.loc4), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, '5. ' + "Add Yossi", tuple(self.loc5), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            # cv2.putText(frame, 'Put fist to go back ', (20, 120), self.font, 1, self.FONT_COLOR, 2, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi, left=left)
            self.add_back_button(frame)
            self.add_renew_button(frame, left)

        elif self.state == self.DRAW:
            cv2.putText(frame, "Draw", tuple(self.loc_TITLE), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            cv2.putText(frame, 'Draw on table', tuple(self.loc1), self.font, self.size, self.FONT_COLOR, self.thick, cv2.LINE_4)
            self.add_back_button(frame, left)
            self.add_renew_button(frame, left)

    def draw_shape_images(self, frame, left: bool = False):
        roi = np.array(self.img_roi, copy=True)
        if not left:
            roi[2] -= self.offset
            roi[3] -= self.offset
        offset = 42
        frame[roi[0]: roi[1], roi[2]: roi[3]] = self.smilie_img
        frame[offset + roi[0]: offset + roi[1], roi[2]: roi[3]] = self.triangle_img
        frame[offset * 2 + roi[0]: offset * 2 + roi[1], roi[2]: roi[3]] = self.circle_img
        frame[offset * 3 + roi[0]: offset * 3 + roi[1], roi[2]: roi[3]] = self.square_img

        diff = 15
        self.loc1[1] = roi[0] + diff
        self.loc2[1] = roi[0] + offset + diff
        self.loc3[1] = roi[0] + offset * 2 + diff
        self.loc4[1] = roi[0] + offset * 3 + diff

    def change_use_locations(self, left: bool = False):
        if left:
            self.loc_TITLE[0] = self.LOC_TITLE[0] + self.offset
            self.loc1[0] = self.LOC1[0] + self.offset
            self.loc2[0] = self.LOC2[0] + self.offset
            self.loc3[0] = self.LOC3[0] + self.offset
            self.loc4[0] = self.LOC4[0] + self.offset
            self.loc5[0] = self.LOC5[0] + self.offset
            self.loc_CENTER_SHAPE[0] = self.LOC_CENTER_SHAPE[0] + self.offset
            self.loc_CENTER_SHAPE_MANI[0] = self.LOC_CENTER_SHAPE_MANI[0] + self.offset
            self.loc_CENTER_SHAPE_INSERT[0] = self.LOC_CENTER_SHAPE_INSERT[0] + self.offset
        else:
            self.loc_TITLE = np.array(self.LOC_TITLE, copy=True)
            self.loc1 = np.array(self.LOC1, copy=True)
            self.loc2 = np.array(self.LOC2, copy=True)
            self.loc3 = np.array(self.LOC3, copy=True)
            self.loc4 = np.array(self.LOC4, copy=True)
            self.loc5 = np.array(self.LOC5, copy=True)
            self.loc_CENTER_SHAPE = np.array(self.LOC_CENTER_SHAPE, copy=True)
            self.loc_CENTER_SHAPE_MANI = np.array(self.LOC_CENTER_SHAPE_MANI, copy=True)
            self.loc_CENTER_SHAPE_INSERT = np.array(self.LOC_CENTER_SHAPE_INSERT, copy=True)

    def write_state_on_frame(self, frame):
        size = 1
        thick = 2
        location = self.loc3
        spaces = "       "
        color = self.COLOR2
        if self.state == self.PAINT_MENU:
            cv2.putText(frame, spaces + "Paint Menu", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.SHAPE_MANIPULATION:
            cv2.putText(frame, "Shape Manipulation", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.CHANGE_COLOR:
            cv2.putText(frame, spaces + "Change Color", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.DELETE_SHAPE:
            cv2.putText(frame, spaces + "Delete Shapes", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.DRAW:
            cv2.putText(frame, spaces + "Draw", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.INSERT_SHAPE:
            cv2.putText(frame, spaces + "Insert Shape", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.TABLET:
            cv2.putText(frame, spaces + "Tablet", tuple(location), self.font, size, color, thick, cv2.LINE_4)
        elif self.state == self.MAIN_MENU:
            cv2.putText(frame, spaces + "Main Menu", tuple(location), self.font, size, color, thick, cv2.LINE_4)

    def add_roi_rectangle(self, frame, roi, text=None, left: bool = False):
        # TODO add text on the Roi
        if left:
            return
        show_roi = np.array(roi, copy=True)
        if left:
            show_roi[2] += self.offset
            show_roi[3] += self.offset
        cv2.rectangle(frame, (show_roi[2], show_roi[0]), (show_roi[3], show_roi[1]), (51, 22, 198), 0)  # (top left corner),(bottom right corner)
        
    def add_back_button(self, frame, left: bool = False):
        if not left:
            return
        roi = np.array(self.back_button_roi, copy=True)
        if not left:
            roi[2] -= self.offset
            roi[3] -= self.offset
        frame[roi[0]: roi[1], roi[2]: roi[3]] = self.back_img

    def add_renew_button(self, frame, left: bool = False):
        if not left:
            return
        roi = np.array(self.renew_table_roi, copy=True)
        if not left:
            roi[2] -= self.offset
            roi[3] -= self.offset
        frame[roi[0]: roi[1], roi[2]: roi[3]] = self.find_table_img

    def state_operations(self, img, mask, application=None):
        if self.state_changed or self.in_delay:  # state just changed - small delay
            if not self.in_delay:  # starts a small delay between states
                self.countdown_clk.start_countdown(self.delay_between_states)
                self.state_changed = False
                self.in_delay = True
            elif self.countdown_clk.check() == 0:
                self.in_delay = False
                self.saved_choice = 0
        else:
            renew_pressed, _ = self.renew_button.check_and_add(mask)
            if self.state != self.MAIN_MENU:
                back_pressed, _ = self.back_button.check_and_add(mask)
                if back_pressed:
                    self.back_button.memory *= 0  # zeros the memory
                    self.count_fingers.memory *= 0
                    self.shape_classifier.memory *= 0
                    self.transition(self.BACK, application)  # TODO add that line

            if renew_pressed:
                self.renew_button.memory *= 0  # zeros the memory
                self.count_fingers.memory *= 0
                self.shape_classifier.memory *= 0
                # self.transition(self.BACK, application)  # TODO add that line
                print('renewed')
                return True  # renew the table segmentation

            elif self.state in [self.MAIN_MENU, self.PAINT_MENU, self.CHANGE_COLOR]:
                num_fingers, fingers_valid = self.count_fingers.check_and_add(mask)
                if fingers_valid:
                    self.count_fingers.memory *= 0  # zeros the memory
                    self.transition(num_fingers, application)  # TODO add that line
            elif self.state == self.INSERT_SHAPE:
                masked_3ch = img * np.concatenate([mask[:, :, np.newaxis] != 0] * 3, axis=2)
                shape_class, shape_valid = self.shape_classifier.check_and_add(masked_3ch)
                if shape_valid:
                    self.shape_classifier.memory *= 0  # zeros the memory
                    self.transition(shape_class, application)  # TODO add that line

        return False   # don't renew the table segmentation


class ClassifierCounter:
    def __init__(self, roi, classifier, memory_size=20, valid_thresh=0.5):
        self.memory = np.zeros(memory_size, dtype=np.int)
        self.idx = 0
        self.memory_size = memory_size
        self.y0, self.y1, self.x0, self.x1 = roi
        self.valid_thresh = valid_thresh
        self.classifier = classifier

    def check_and_add(self, img):
        img_in_roi = img[self.y0: self.y1, self.x0: self.x1]
        if (img_in_roi != 0).sum()/img_in_roi.size > 0.1:  # enough hands in roi to calculate # TODO adjust
            # calculate the num of fingers/the svm class of the img
            label = self.classifier(img)  # TODO finger counter gets roi only??
        else:
            label = 0
        print(label)
        self.memory[self.idx] = label
        self.idx = (self.idx + 1) % self.memory_size
        common_label = np.bincount(self.memory).argmax()
        repetitions = (self.memory == common_label).sum()
        valid = ((repetitions/self.memory.size) > self.valid_thresh) and common_label != 0
        return common_label, valid


class Button:
    def __init__(self, roi, name, memory_size=12, valid_thresh=0.7):
        self.name = name
        self.memory = np.zeros(memory_size, dtype=np.int)
        self.idx = 0
        self.memory_size = memory_size
        self.y0, self.y1, self.x0, self.x1 = roi
        self.valid_thresh = valid_thresh

    def check_and_add(self, img):
        img_in_roi = img[self.y0: self.y1, self.x0: self.x1]
        if (img_in_roi != 0).sum()/img_in_roi.size > 0.5:  # enough hands in roi to calculate # TODO adjust
            # calculate the num of fingers/the svm class of the img
            label = 1
        else:
            label = 0
        self.memory[self.idx] = label
        self.idx = (self.idx + 1) % self.memory_size
        common_label = np.bincount(self.memory).argmax()
        repetitions = (self.memory == common_label).sum()
        valid = ((repetitions/self.memory.size) > self.valid_thresh) and common_label != 0
        return common_label, valid


if __name__ == "__main__":
    from HandOperations import HandOperations
    from aux_functions import *

    class DummyApp:
        PAINT = 1
        TABLET = 2
        INSERT_SHAPE = 3
        DELETE = 4
        EXIT_DELETE = 5
        CHANGE_COLOR = 6
        DRAW = 7
        EXIT_DRAW = 8
        def __init__(self):
            self.enable_touch = False

        def run(self, app_name):
            print(f"running: {app_name}")

        def exit(self):
            print(f"exiting")

        def change_paint_state(self, state, arg=1):
            print(f"changing paint state to: {state}, with arg: {arg}")

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(1)
    hand_operations = HandOperations()
    application = DummyApp()
    state_machine = StateMachine(hand_operations, cap.read()[1].shape[: 2], crop_x=0)
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    # Read until video is completed

    # count_fingers = ClassifierCounter(state_machine.classifier_roi,
    #                                   classifier=hand_operations.get_finger_num)
    # shape_classifier = ClassifierCounter(state_machine.classifier_roi,
    #                                      classifier=hand_operations.predict_shape)
    # back_button = BackButton(state_machine.back_button_roi)
    # countdown_clk = CountDown()

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        im_l = frame  # cv2.flip(frame, 1)
        im_r = np.array(frame, copy=True)
        mask_l = hand_operations.get_hand_mask(im_l)
        state_machine.state_operations(im_l, mask_l, application)


        #
        # if state_machine.state_changed or state_machine.in_delay:  # state just changed - small delay
        #     if not state_machine.in_delay:  # starts a small delay between states
        #         countdown_clk.start_countdown(state_machine.delay_between_states)
        #         state_machine.state_changed = False
        #         state_machine.in_delay = True
        #     elif countdown_clk.check() == 0:
        #         state_machine.in_delay = False
        #     print("in delay")
        # else:
        #     if state_machine.state != self.MAIN_MENU:
        #         back_pressed, _ = back_button.check_and_add(mask_l)
        #         if back_pressed:
        #             print(f"back pressed !!")
        #             back_button.memory *= 0  # zeros the memory
        #             count_fingers.memory *= 0
        #             shape_classifier.memory *= 0
        #             state_machine.transition(state_machine.BACK)  # TODO add that line
        #     if state_machine.state in [self.MAIN_MENU, self.PAINT_MENU, self.CHANGE_COLOR]:
        #         num_fingers, fingers_valid = count_fingers.check_and_add(mask_l)
        #         if fingers_valid:
        #             print(f"num_fingers: {num_fingers}")
        #             count_fingers.memory *= 0  # zeros the memory
        #             state_machine.transition(num_fingers)  # TODO add that line
        #     elif state_machine.state == self.INSERT_SHAPE:
        #         masked_3ch = im_l * np.concatenate([mask_l[:, :, np.newaxis] != 0] * 3, axis=2)
        #         shape_class, shape_valid = shape_classifier.check_and_add(masked_3ch)
        #         if shape_valid:
        #             print(f"shape_class: {shape_class}")
        #             shape_classifier.memory *= 0  # zeros the memory
        #             state_machine.transition(shape_class)  # TODO add that line

        # key = cv2.waitKey(30)
        # transition_done = state_machine.transition(key)
        # if transition_done:
        #     print("transition")

        state_machine.add_text(im_l, left=True)
        # state_machine.add_text(im_r, left=False)

        #Display the resulting frame
        cv2.imshow('Frame Left', im_l)
        # cv2.imshow('Frame Right', im_r)
        cv2.imshow('mask', mask_l)

        # Press Q on keyboard to  exit
        if cv2.waitKey(80) & 0xFF == ord('q'):
          break

    # When everything done, release the video capture object
    cap.release()

