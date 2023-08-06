# IMP_AR_project

https://www.youtube.com/watch?v=I7b0lHkV0Ug&t=1s

## Overview

Augmented reality (AR) is one of the most popular technologies, and the tech giants such as Apple, Meta, Google, and more invest hundreds of millions. These days AR still haven't made a breakthrough, and we decided to join the challenge.
We created an AR application with a stereo camera, PC, and mobile phone that transforms any table into a tablet or gamepad.

![AR](Images/ar.png)

## Project Documentation

For an explanation of the project, there is [documentation](Documantation/AR_pdf.pdf).

### Social Media Posts

[Medium](https://medium.com/@ophirg123/ar-project-convert-any-table-to-a-tablet-86b854eaa4df)  
[Youtube](https://www.youtube.com/watch?v=I7b0lHkV0Ug)  
[Linkedin](https://www.linkedin.com/posts/ophir-gruteke-computer-engineering_ar-project-convert-any-table-to-a-tablet-activity-6914210993573105665-boj-?utm_source=linkedin_share&utm_medium=member_desktop_web)

## Hardware

The leading hardware that we used in the project:
- ZED stereo camera
- Nvidia Xavier (Capture frames in high FPS)
- i7 CPU PC (Without GPU)
- Samsung Galaxy S21
- Samsung Tab S7 

![whole_image](Images/integrate.png)

## Perquisites

- Opencv-python 4.5 compiled with gstreamer
- Android Studio
  - Download [Gstreamer android tutorial](https://gstreamer.freedesktop.org/documentation/tutorials/android/video.html).
  - Replace the files from Android folder inside tutorial 3.
  - Install the app on the phone.
- ZED sdk installed on the Jetson Xavier.
- [Scrcpy](https://github.com/Genymobile/scrcpy)
## Run the project

1. Connect the camera to the jetson and run the file Zed\zed_opencv.py.
2. Connect the tablet to the PC and run the software Scrcpy to get the tablet screen.
3. Run the file Main\main.py   
Make sure that you changes all the IP's as the local network.
