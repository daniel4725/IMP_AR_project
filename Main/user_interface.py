import cv2
import numpy as np
import pyautogui
import mouse
# import d3dshot
import win32gui

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
    def __init__(self, map, left_up_corner=(44, 255)):
        # use pyautogui.position() to get the current position
        self.x, self.width = left_up_corner[0], map.shape[1]
        self.y, self.height = left_up_corner[1], map.shape[0]
        self.touchscreen = TouchScreen()
        # self.screenshoter = d3dshot.create(capture_output="numpy")
        self.region = (self.x, self.y, self.x + self.width, self.y + self.height)

        tablet_window = [(hwnd, title) for hwnd, title in winlist if 'Tablet' in title][0]
        self.hwnd = tablet_window[0]
        win32gui.MoveWindow(self.hwnd, 0, 0, 1000, 1000, True)

        # self.region = win32gui.GetWindowRect(self.hwnd)

    def get_whole_map(self):
        # read the screenshot and convert to cv2 BGR
        return cv2.cvtColor(self.screenshoter.screenshot(region=self.region), cv2.COLOR_RGB2BGR)

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
    def __init__(self, map, duration=0.3, left_up_corner=(44, 255)):
        # use pyautogui.position() to get the current position
        self.x, self.width = left_up_corner[0], map.shape[1]
        self.y, self.height = left_up_corner[1], map.shape[0]
        self.touchscreen = TouchScreen()
        # self.screenshoter = d3dshot.create(capture_output="numpy")
        self.region = (self.x, self.y, self.x + self.width, self.y + self.height)
        self.delete_state = False

        paint_window = [(hwnd, title) for hwnd, title in winlist if 'PaintApp' in title][0]
        self.hwnd = paint_window[0]
        win32gui.MoveWindow(self.hwnd, 0, 0, 1000, 1000, True)

        bbox = win32gui.GetWindowRect(self.hwnd)

        self.duration = duration
        self.buttons_loc_dict = {"insert_shape": (769, 155), "home": (1794, 73),
                                "paint": (1613, 70), "draw_pen": (1654, 135),
                                "normal_mouse": (1873, 143), "change_color": (None, None),

                                "circle": (543, 258), "triangle": (484, 254),
                                "rectangle": (574, 253), "smiley": (694, 257)}

        self.colors_dict = {"red": (None, None), "blue": (None, None), "green": (None, None),
                            "yellow": (None, None)}

    def get_whole_map(self):
        # read the screenshot and convert to cv2 BGR
        return cv2.cvtColor(self.screenshoter.screenshot(region=self.region), cv2.COLOR_RGB2BGR)

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
            if self.delete_state:
                pyautogui.press('delete')
        elif clicked_before and (not clicked_now):  # released click
            mouse.release("left")
        elif clicked_before and clicked_now:  # the mouse is down already - just move it
            dx, dy = self.touchscreen.click_location
            mouse.move(self.x + dx, self.y + dy)

    def insert_shape(self, shape):
        print("insert_shape"); return
        x, y = self.buttons_loc_dict["home"]
        mouse.move(x, y)
        mouse.click("left")  # click - home

        x, y = self.buttons_loc_dict["insert_shape"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left") # click - insert shape

        x, y = self.buttons_loc_dict[shape]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left") # # click - the chosen shape

    def insert_draw(self):
        print("insert_draw"); return
        x, y = self.buttons_loc_dict["paint"]  # click - paint
        mouse.move(x, y)
        mouse.click("left")

        x, y = self.buttons_loc_dict["normal_mouse"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

        x, y = self.buttons_loc_dict["draw_pen"]
        mouse.move(x, y, duration=self.duration)
        mouse.click("left")

    def change_color(self, color):
        print("change_color"); return

    def move_to_front(self):
        win32gui.SetForegroundWindow(self.hwnd)


class Application:
    def __init__(self, map):
        self.none_map = map.copy()
        corners = [(0, 0), (map.shape[1], 0), (map.shape[1], map.shape[0]), (0, map.shape[0])]
        for i in range(4):
            start_point = corners[i]
            end_point = corners[(i + 1) % 4]
            color = (1, 1, 1)
            thickness = 15
            cv2.line(self.none_map, start_point, end_point, color, thickness)
            cv2.line(self.none_map, start_point, end_point, (170, 0, 170), 4 * thickness//5)
            cv2.line(self.none_map, start_point, end_point, color, thickness//5)

        self.paint_app = PaintApp(map)
        self.tablet = TabletApp(map)
        self.running_app = None

    def run(self, app_name):
        if app_name == 'paint':
            self.running_app = self.paint_app
            self.running_app.move_to_front()
        elif app_name == 'tablet':
            self.running_app = self.tablet
            self.running_app.move_to_front()

    def exit(self):
        self.running_app = None

    def get_whole_map(self):
        if self.running_app is None:
            return self.none_map
        else:
            map = self.running_app.get_whole_map()
            map[self.none_map != 0] = self.none_map[self.none_map != 0]
            return map

    def update_touches(self, touch_idx, real_mouse_clicks=True):
        if self.running_app is None:
            return
        else:
            self.running_app.update(touch_idx, real_mouse_clicks)

    def change_paint_state(self, state, arg=1):
        if not isinstance(self.running_app, PaintApp):
            print('\x1b[0;30;41m' + "change_paint_state in application  went wrong!!" + '\x1b[0m')
            assert()
        if state == 'insert_shape':
            self.paint_app.insert_shape(arg)
        if state == 'insert_draw':
            self.paint_app.insert_draw()
        if state == 'change_color':
            self.paint_app.change_color(arg)
        if state == 'delete':
            self.paint_app.delete_state = True
        if state == 'exit_delete':
            self.paint_app.delete_state = False

if __name__ == "__main__":
    map = np.zeros((300, 600))
    app = Application(map)
    print(1)


