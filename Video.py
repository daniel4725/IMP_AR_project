import cv2

class Video_operations:
    
    def __init__(self):
        pass
    
    def view_video_from_path(self, path: str):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        # get total number of frames
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        suc, prev = cap.read()
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        # out.write(prevgray-prevgray)

        for image in range(int(totalFrames) - 1):

            suc, img = cap.read()

            cv2.imshow('Video', img)

            key = cv2.waitKey(int(500 / fps))
            if key == ord('q'):
                break
  
        cap.release()
        cv2.destroyAllWindows()
    
    def save_and_preview_video_from_other_video(self, func, source: str, destination: str, ):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(destination, -1, 20.0, (640,480))
        # get total number of frames
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        for image in range(int(totalFrames)):

            suc, img = cap.read()
               
            func_img = func(img)
            dim = (640, 480)
            func_img_resized = cv2.resize(func_img, dim, interpolation = cv2.INTER_AREA)
            imagenorm = cv2.normalize(func_img_resized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            out.write(imagenorm)
            cv2.imshow('Video', func_img)
            cv2.imshow('Original', img)

            key = cv2.waitKey(100)
            if key == ord('q'):
                break

            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    
    video = Video_operations()
    # video.view_video_from_path("regular_video_right.mp4")

    from Calibration import Calibration
    
    cal = Calibration()
    cal.load_saved_model('hand_gmm_model.sav')
    cal.load_saved_best_labels('hand_best_labels.sav')
    
    from HandOperations import HandOperations
    
    def get_hand_mask(image):
        hand = HandOperations(image=image)
        masked_image = hand.get_hand_mask(cal.get_segmentation(image))
        # count_image = hand.finger_count(masked_image)
        return masked_image
    
    video.save_and_preview_video_from_other_video(get_hand_mask, "regular_video_left.mp4", "masked_regular_video_left.mp4")
        