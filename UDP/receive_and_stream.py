import os
import sys

os.add_dll_directory(r'C:\opencv43\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\bin')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\lib\gstreamer-1.0')
os.add_dll_directory(r'C:\gstreamer\1.0\msvc_x86_64\lib')

import numpy as np 
import cv2 
from multiprocessing import Process


def send():
    # cap_send = cv2.VideoCapture('videotestsrc ! video/x-raw,framerate=20/1 ! videoscale ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    cap_send = cv2.VideoCapture("regular_video.mp4")

    out_send = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=192.168.0.169 port=5005',cv2.CAP_GSTREAMER,0, 20, (1280,360), True)

    if not cap_send.isOpened() or not out_send.isOpened():
        print('VideoCapture or VideoWriter not opened')
        exit(0)

    while True:
        ret,frame = cap_send.read()

        if not ret:
            print('empty frame')
            break

        out_send.write(frame)

        cv2.imshow('send', frame)
        if cv2.waitKey(100)&0xFF == ord('q'):
            break

    cap_send.release()
    out_send.release()

def receive():
    cap_receive = cv2.VideoCapture('udpsrc port=5005 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

    out_send = cv2.VideoWriter('appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=192.168.0.169 port=5005',cv2.CAP_GSTREAMER,0, 20, (1280,360), True)

    
    if not cap_receive.isOpened():
        print('VideoCapture not opened')
        exit(0)

    while True:
        ret,frame = cap_receive.read()

        if not ret:
            print('empty frame')
            break
        
        out_send.write(frame)

        cv2.imshow('receive', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    #cap_receive.release()

if __name__ == '__main__':
    # s = Process(target=send)
    r = Process(target=receive)
    # s.start()
    r.start()
    # s.join()
    r.join()

    cv2.destroyAllWindows()