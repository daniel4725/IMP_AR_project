from PIL import ImageGrab
import cv2
import win32gui
import win32api
import win32con
import numpy as np 
import time

toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
win32gui.EnumWindows(enum_cb, toplist)

tablet = [(hwnd, title) for hwnd, title in winlist if 'SM-T860' in title]
tablet = tablet[0]
hwnd = tablet[0]

win32gui.SetForegroundWindow(hwnd)
win32gui.MoveWindow(hwnd, 50, 50, 800, 600, True)
win32api.SetCursorPos([453,588])
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
time.sleep(0.01)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
time.sleep(0.01)
win32api.SetCursorPos([392,464])
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
time.sleep(0.01)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
time.sleep(0.01)
bbox = win32gui.GetWindowRect(hwnd)
img = ImageGrab.grab(bbox)

# img.show()

while(1):
    print(win32api.GetCursorPos())
    time.sleep(1)
    

while(1):
        img = ImageGrab.grab(bbox)
        img = np.array(img)
        cv2.imshow("test", img)
        
        key = cv2.waitKey(5)
        if key == ord('q'):
            break