# import libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear

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


server = NetGear() #Define netgear server with default settings

# infinite loop until [Ctrl+C] is pressed
while True:
    try: 
        img = ImageGrab.grab(bbox)
        frame = np.array(img)

        # read frames

        # check if frame is None
        if frame is None:
            #if True break the infinite loop
            break

        # do something with frame here

        # send frame to server
        server.send(frame)
    
    except KeyboardInterrupt:
        #break the infinite loop
        break

# safely close video stream
# safely close server
server.close()