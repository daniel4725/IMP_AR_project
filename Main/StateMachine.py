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

    def __init__(self, hand_operations, shape):
        self.state = self.MAIN_MENU
        self.former_state = self.MAIN_MENU
        self.text = ""
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.classifier_roi = [140, 360, 400, 620]  # [y_start, y_end, x_start, x_end]
        back_roi = [0, 40, shape[1] - 40, shape[1]]
        back_dim = (back_roi[3] - back_roi[2], back_roi[1] - back_roi[0])
        self.back_button_roi = back_roi
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        back_img = cv2.imread(os.path.join(directory, "back_button.jpg"))
        self.back_img = cv2.resize(back_img, back_dim, interpolation=cv2.INTER_AREA)
        self.in_delay = False
        self.delay_between_states = 2
        self.state_changed = False
        self.saved_choice = 0

        self.count_fingers = ClassifierCounter(self.classifier_roi, classifier=hand_operations.get_finger_num)
        self.shape_classifier = ClassifierCounter(self.classifier_roi, classifier=hand_operations.predict_shape)
        self.back_button = BackButton(self.back_button_roi)
        self.countdown_clk = CountDown()

        self.shape_name_dict = {self.TRIANGLE: "triangle", self.CIRCLE: "circle",
                                self.SQUARE: "square", self.SMILIE: "smilie"}

        self.color_name_dict = {self.BLUE: "blue", self.GREEN: "green", self.RED: "red",
                                self.YELLOW: "yellow", self.PURPLE: "purple"}

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

    def add_text(self, frame):
        if self.in_delay or self.state_changed:
            if (self.state in [self.SHAPE_MANIPULATION, self.PAINT_MENU]) and (self.saved_choice != 0):
                cv2.putText(frame, self.saved_choice, (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
            else:
                self.write_state_on_frame(frame)

        elif self.state == self.MAIN_MENU:
            cv2.putText(frame, '1. Paint app', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '2. Tablet', (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi)
        elif self.state == self.PAINT_MENU:
            cv2.putText(frame, '1. Insert shape', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '2. Delete shape', (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '3. Shape Manipulation', (20, 60), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '4. Change color', (20, 80), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '5. Draw', (20, 100), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, 'Put fist to go back ', (20, 120), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi)
            self.add_back_button(frame)

        elif self.state == self.TABLET:
            cv2.putText(frame, 'Put fist to go back', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_back_button(frame)

        elif self.state == self.INSERT_SHAPE:
            cv2.putText(frame, 'Draw shape with hands', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, 'Put fist to go back ', (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi)
            self.add_back_button(frame)

        elif self.state == self.DELETE_SHAPE:
            cv2.putText(frame, 'Touch shape to delete', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, 'Put fist to go back ', (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_back_button(frame)

        elif self.state == self.SHAPE_MANIPULATION:
            cv2.putText(frame, 'Touch shape to move', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, 'Put fist to go back ', (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_back_button(frame)

        elif self.state == self.CHANGE_COLOR:
            cv2.putText(frame, '1. ' + self.color_name_dict[1], (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '2. ' + self.color_name_dict[2], (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '3. ' + self.color_name_dict[3], (20, 60), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '4. ' + self.color_name_dict[4], (20, 80), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, '5. ' + self.color_name_dict[5], (20, 100), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, 'Put fist to go back ', (20, 120), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_roi_rectangle(frame, self.classifier_roi)
            self.add_back_button(frame)

        elif self.state == self.DRAW:
            cv2.putText(frame, 'Draw on table', (20, 20), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            cv2.putText(frame, 'Put fist to go back ', (20, 40), self.font, 1, (125, 125, 0), 2, cv2.LINE_4)
            self.add_back_button(frame)

    def write_state_on_frame(self, frame):
        if self.state == self.PAINT_MENU:
            cv2.putText(frame, "Paint Menu", (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.SHAPE_MANIPULATION:
            cv2.putText(frame, "Shape Manipulation", (100, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.CHANGE_COLOR:
            cv2.putText(frame, "Change Color", (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.DELETE_SHAPE:
            cv2.putText(frame, "Delete Shapes", (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.DRAW:
            cv2.putText(frame, "Draw", (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.INSERT_SHAPE:
            cv2.putText(frame, "Insert Shape", (150, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.TABLET:
            cv2.putText(frame, "Tablet", (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)
        elif self.state == self.MAIN_MENU:
            cv2.putText(frame, "Main Menu", (200, 200), self.font, 3, (125, 125, 0), 4, cv2.LINE_4)

    def add_roi_rectangle(self, frame, roi, text=None):
        # TODO add text on the Roi
        cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)
        
    def add_back_button(self, frame):
        roi = self.back_button_roi
        frame[roi[0]: roi[1], roi[2]: roi[3]] = self.back_img

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
            if self.state != self.MAIN_MENU:
                back_pressed, _ = self.back_button.check_and_add(mask)
                if back_pressed:
                    self.back_button.memory *= 0  # zeros the memory
                    self.count_fingers.memory *= 0
                    self.shape_classifier.memory *= 0
                    self.transition(self.BACK, application)  # TODO add that line
            if self.state in [self.MAIN_MENU, self.PAINT_MENU, self.CHANGE_COLOR]:
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
        self.memory[self.idx] = label
        self.idx = (self.idx + 1) % self.memory_size
        common_label = np.bincount(self.memory).argmax()
        repetitions = (self.memory == common_label).sum()
        valid = ((repetitions/self.memory.size) > self.valid_thresh) and common_label != 0
        return common_label, valid


class BackButton:
    def __init__(self, roi, memory_size=20, valid_thresh=0.7):
        self.memory = np.zeros(memory_size, dtype=np.int)
        self.idx = 0
        self.memory_size = memory_size
        self.y0, self.y1, self.x0, self.x1 = roi
        self.valid_thresh = valid_thresh

    def check_and_add(self, img):
        img_in_roi = img[self.y0: self.y1, self.x0: self.x1]
        if (img_in_roi != 0).sum()/img_in_roi.size > 0.9:  # enough hands in roi to calculate # TODO adjust
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
    state_machine = StateMachine(hand_operations, cap.read()[1].shape[: 2])
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

        state_machine.add_text(im_l)

        #Display the resulting frame
        cv2.imshow('Frame', im_l)
        cv2.imshow('mask', mask_l)

        # Press Q on keyboard to  exit
        if cv2.waitKey(80) & 0xFF == ord('q'):
          break

    # When everything done, release the video capture object
    cap.release()

