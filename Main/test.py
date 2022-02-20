import os
import sys

import pickle
from table_handling import CornersFollower
from PIL import ImageGrab
import cv2
import win32gui
import win32api
import win32con
import numpy as np
import time
from user_interface import Application
import matplotlib.pyplot as plt


app = Application(np.zeros((300, 500, 3)))
app.run(app.PAINT)
time.sleep(1)
img = app.get_whole_map()
plt.imshow(img)
plt.show()
a = 5

# toplist, winlist = [], []
# def enum_cb(hwnd, results):
#     winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
# win32gui.EnumWindows(enum_cb, toplist)
#
# tablet = [(hwnd, title) for hwnd, title in winlist if 'Tablet' in title]
# tablet = tablet[0]
# hwnd = tablet[0]
#
# bbox = win32gui.GetWindowRect(hwnd)
# img = ImageGrab.grab(bbox)
