
from multiprocessing import Process

from Video import Video_operations

from functools import wraps
from time import time

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it


def test_function(frame):
    return frame

def main():
    video = Video_operations()
    
    gstreamer_writer = video.open_gstreamer_video_writer()
    gstreamer_capture = video.open_gstreamer_video_capture()
    
    video.view_and_send_video(gstreamer_capture, gstreamer_writer, test_function)
    video.close_gstreamer_video_writer()
    video.close_gstreamer_video_capture()
    
    
    
if __name__ == "__main__":
    main()