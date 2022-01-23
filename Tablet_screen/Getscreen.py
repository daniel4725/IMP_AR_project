from PIL import ImageGrab
import cv2
import win32gui
import numpy as np 

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

tablet = [(hwnd, title) for hwnd, title in winlist if 'SM-T860' in title]
tablet = tablet[0]
hwnd = tablet[0]

win32gui.SetForegroundWindow(hwnd)
bbox = win32gui.GetWindowRect(hwnd)
img = ImageGrab.grab(bbox)

# img.show()

while(1):
        img = ImageGrab.grab(bbox)
        img = np.array(img)
        cv2.imshow("test", img)
        
        key = cv2.waitKey(5)
        if key == ord('q'):
            break