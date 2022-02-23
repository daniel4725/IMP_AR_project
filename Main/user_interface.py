import cv2
import numpy as np
import pyautogui
import mouse
import win32gui
from PIL import ImageGrab


toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)


class TouchScreen:
    def __init__(self, click_sustain=3, max_clicks_dist=50): # TODO use the max dist or not?
        # TODO doc.   click_sustain: the number of updates that a click stays clicked (sustained) after no new touches
        self.clicked = False
        self.click_location = np.zeros([0, 2])
        self.click_sustain = click_sustain
        self.current_sustain = 0
        self.max_clicks_dist = max_clicks_dist # TODO use the max dist or not?

    def update(self, touch_idx):
        if touch_idx.shape[0] == 0:  # no touches
            if self.current_sustain == 0:  # if the click does not needs to be sustained
                if self.clicked:
                    self.click_location = np.zeros([0, 2])
                    self.clicked = False
            else:   # if needs to sustain do nothing - keep clicked and decrement sustain
                self.current_sustain -= 1
        else:  # touch with one finger
            self.clicked = True
            current_location = touch_idx[0]  # takes the closest touch
            # distance = np.linalg.norm(self.click_location - current_location)  # TODO use the max dist or not?
            self.click_location = current_location
            self.current_sustain = self.click_sustain   # reset the click sustain counter


class TabletApp:
    def __init__(self, map):
        upper_bar_size = 37
        sides_size = 5
        self.touchscreen = TouchScreen()
        self.x, self.width = sides_size, map.shape[1]
        self.y, self.height = upper_bar_size, map.shape[0]
        self.touchscreen = TouchScreen()
        self.region = (self.x, self.y, self.x + self.width, self.y + self.height)

        # tablet_window = [(hwnd, title) for hwnd, title in winlist if 'SM-P610' in title][0]
        tablet_window = [(hwnd, title) for hwnd, title in winlist if 'SM-T860' in title][0]
        # tablet_window = [(hwnd, title) for hwnd, title in winlist if 'Tablet' in title][0]
        self.hwnd = tablet_window[0]
        win32gui.MoveWindow(self.hwnd, 0, 0, map.shape[1] + 2 * sides_size, map.shape[0] + sides_size + upper_bar_size, True)
        # ScreenToClient
        # GetWindowRect
        # self.region = win32gui.ScreenToClient(self.hwnd)


    def update(self, touch_idx, real_mouse_clicks=True):
        clicked_before = self.touchscreen.clicked
        self.touchscreen.update(touch_idx)
        clicked_now = self.touchscreen.clicked
        if not real_mouse_clicks:  # for debugging - don't perform real clicks
            return
        if (not clicked_before) and clicked_now:  # there was a click now
            dx, dy = self.touchscreen.click_location
            mouse.move(self.x + dx, self.y + dy)
            mouse.press("left")
        elif clicked_before and (not clicked_now):  # released click
            mouse.release("left")
        elif clicked_before and clicked_now:  # the mouse is down already - just move it
            dx, dy = self.touchscreen.click_location
            mouse.move(self.x + dx, self.y + dy)

    def move_to_front(self):
        win32gui.SetForegroundWindow(self.hwnd)


class PaintApp:
    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4
    PURPLE = 5

    TRIANGLE = 1
    CIRCLE = 2
    SQUARE = 3
    SMILIE = 4
    YOSSI = 5

    def __init__(self, map, duration=0.25, left_up_corner=(41, 279)):
        # import pyautogui
        # use pyautogui.position() to get the current position
        self.x, self.width = left_up_corner[0], map.shape[1]
        self.y, self.height = left_up_corner[1], map.shape[0]
        self.touchscreen = TouchScreen()
        self.region = (self.x, self.y, self.x + self.width, self.y + self.height)
        self.delete_state = False

        paint_window = [(hwnd, title) for hwnd, title in winlist if 'PaintApp' in title][0]
        self.hwnd = paint_window[0]


        self.duration = duration
        self.buttons_loc_dict = {"insert_shape": (1150, 175), "home": (122, 91),
                                "paint": (280, 94), "draw_pen": (287, 177),
                                "normal_mouse": (56, 162), "change_color": (1375, 142),

                                self.CIRCLE: (1167, 580), self.TRIANGLE: (1195, 573),
                                self.SQUARE: (1133, 487), self.SMILIE: (1281, 637),

                                 "default_shape": (1562, 870), self.YOSSI: (1565, 711)}

        self.shape_colors_dict = {self.BLUE: (1499, 418), self.GREEN: (1473, 419), self.RED: (1344, 420),
                                  self.YELLOW: (1426, 422), self.PURPLE: (1579, 419)}
        self.pen_colors_dict = {self.BLUE: (327, 672), self.GREEN: (430, 673), self.RED: (438, 568),
                                self.YELLOW: (281, 563), self.PURPLE: (275, 619)}

    def update(self, touch_idx, real_mouse_clicks=True):
        clicked_before = self.touchscreen.clicked
        self.touchscreen.update(touch_idx)
        clicked_now = self.touchscreen.clicked
        if not real_mouse_clicks:  # for debugging - don't perform real clicks
            return
        if (not clicked_before) and clicked_now:  # there was a click now
            dx, dy = self.touchscreen.click_location
            mouse.move(self.x + dx, self.y + dy)
            mouse.press("left")
        elif clicked_before and (not clicked_now):  # released click
            mouse.release("left")
            if self.delete_state:
                pyautogui.press('delete')
        elif clicked_before and clicked_now:  # the mouse is down already - just move it
            dx, dy = self.touchscreen.click_location
            mouse.move(self.x + dx, self.y + dy)

    def insert_shape(self, shape):
        x, y = self.buttons_loc_dict["home"]
        mouse.move(x, y)
        mouse.click("left")  # click - home

        x, y = self.buttons_loc_dict["insert_shape"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")  # click - insert shape

        x, y = self.buttons_loc_dict[shape]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")  # click - the chosen shape

        mouse.move(self.x + self.width//4, self.y + self.height//4, duration=self.duration)
        mouse.click("left")  # insert the shape

    def insert_draw(self):
        x, y = self.buttons_loc_dict["paint"]  # click - paint
        mouse.move(x, y)
        mouse.click("left")

        x, y = self.buttons_loc_dict["normal_mouse"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

        x, y = self.buttons_loc_dict["draw_pen"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

    def exit_draw(self):
        x, y = self.buttons_loc_dict["paint"]  # click - paint
        mouse.move(x, y)
        mouse.click("left")

        x, y = self.buttons_loc_dict["normal_mouse"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

    def change_color(self, color):
        self.change_shape_color(color)
        self.change_pen_color(color)
        x, y = self.buttons_loc_dict["home"]
        mouse.move(x, y)
        mouse.click("left")  # click - home

    def change_shape_color(self, color):
        x, y = self.buttons_loc_dict["home"]
        mouse.move(x, y)
        mouse.click("left")  # click - home
        x, y = self.buttons_loc_dict["default_shape"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")  # click the default shape
        x, y = self.buttons_loc_dict["change_color"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")
        x, y = self.shape_colors_dict[color]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

        # set the default shapes to be that color
        x, y = self.buttons_loc_dict["default_shape"]  # right click the default shape
        mouse.move(x, y, duration=self.duration)
        mouse.click("right")
        pyautogui.press('d')

    def change_pen_color(self, color):
        x, y = self.buttons_loc_dict["paint"]  # click - paint
        mouse.move(x, y)
        mouse.click("left")

        x, y = self.buttons_loc_dict["draw_pen"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")
        mouse.click("left")

        x, y = self.pen_colors_dict[color]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

        x, y = self.buttons_loc_dict["normal_mouse"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

    def insert_yossi(self):
        x, y = self.buttons_loc_dict[self.YOSSI]
        mouse.move(x, y)
        mouse.press("left")
        pyautogui.keyDown('ctrl')
        mouse.move(self.x + self.width//4, self.y + self.height//4, duration=self.duration)
        mouse.release("left")
        pyautogui.keyUp('ctrl')

    def move_to_front(self):
        win32gui.SetForegroundWindow(self.hwnd)


class Application:
    PAINT = 1
    TABLET = 2
    INSERT_SHAPE = 3
    DELETE = 4
    EXIT_DELETE = 5
    CHANGE_COLOR = 6
    DRAW = 7
    EXIT_DRAW = 8

    BLUE = 1
    GREEN = 2
    RED = 3
    YELLOW = 4
    PURPLE = 5

    TRIANGLE = 1
    CIRCLE = 2
    SQUARE = 3
    SMILIE = 4
    YOSSI = 5

    def __init__(self, map):
        self.none_map = map.copy()
        corners = [(0, 0), (map.shape[1], 0), (map.shape[1], map.shape[0]), (0, map.shape[0])]
        for i in range(4):
            start_point = corners[i]
            end_point = corners[(i + 1) % 4]
            color1 = (1, 1, 1)
            color2 = (170, 1, 170)
            thickness = 15
            cv2.line(self.none_map, start_point, end_point, color1, thickness)
            cv2.line(self.none_map, start_point, end_point, color2, 4 * thickness//5)
            cv2.line(self.none_map, start_point, end_point, color1, thickness//5)

        self.paint_app = PaintApp(map)
        self.tablet = TabletApp(map)
        self.running_app = None
        self.enable_touch = False
        self.allow_clicks = True    # TODO make true

    def run(self, app_name):
        if app_name == self.PAINT:
            self.running_app = self.paint_app
            self.running_app.move_to_front()
        elif app_name == self.TABLET:
            self.running_app = self.tablet
            self.running_app.move_to_front()

    def exit(self):
        self.running_app = None

    def get_whole_map(self):
        if self.running_app is None:
            return self.none_map
        else:
            map = ImageGrab.grab(self.running_app.region)
            map = cv2.cvtColor(np.array(map, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            map[self.none_map != 0] = self.none_map[self.none_map != 0]
            if isinstance(self.running_app, TabletApp):
                map += 1
            return map

    def update_touches(self, touch_idx):
        if self.running_app is None:
            return
        else:
            self.running_app.update(touch_idx, self.enable_touch and self.allow_clicks)

    def change_paint_state(self, operation, arg=1):
        if not isinstance(self.running_app, PaintApp):
            print('\x1b[0;30;41m' + "change_paint_state in application  went wrong!!" + '\x1b[0m')
            assert()
        if operation == self.INSERT_SHAPE:
            self.paint_app.insert_shape(arg)
        if operation == self.DRAW:
            self.paint_app.insert_draw()
        if operation == self.EXIT_DRAW:
            self.paint_app.exit_draw()
        if operation == self.CHANGE_COLOR:
            if arg == self.PURPLE:
                self.paint_app.insert_yossi()
            else:
                self.paint_app.change_color(arg)
        if operation == self.DELETE:
            self.paint_app.delete_state = True
        if operation == self.EXIT_DELETE:
            self.paint_app.delete_state = False

if __name__ == "__main__":
    import time
    map = np.zeros((600, 1200, 3))
    app = Application(map)
    app.run(app.PAINT)
    time.sleep(0.1)
    # cv2.imshow("Ww", app.get_whole_map())
    # cv2.waitKey(0)
    app.paint_app.insert_yossi()

    app.change_paint_state(app.CHANGE_COLOR, app.GREEN)
    app.change_paint_state(app.INSERT_SHAPE, app.SQUARE)
    # app.paint_app.insert_yossi()
    print(1)


