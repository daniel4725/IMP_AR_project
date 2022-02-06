import sys
import numpy as np
import pyzed.sl as sl
import cv2

from Calibration import Calibration
from HandOperations import HandOperations


def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.camera_fps = 15

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed_left = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    image_zed_right = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    key = ' '
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    masked_out = cv2.VideoWriter('masked_video.mp4', fourcc, 5.0, (image_size.width*2,image_size.height))
    regular_left_out = cv2.VideoWriter('regular_video_left.mp4', fourcc, 5.0, (image_size.width,image_size.height))
    regular_right_out = cv2.VideoWriter('regular_video_right.mp4', fourcc, 5.0, (image_size.width,image_size.height))

                     
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv_left = image_zed_left.get_data()
            image_ocv_right = image_zed_right.get_data()
            
            # img = image_ocv_left[:, 0:int(image_ocv_left.shape[1] / 2), :]  # left image
            hand_left = HandOperations(image=image_ocv_left)
            masked_image_left = hand_left.get_hand_mask(gmm_model=load_model)
            hand_right = HandOperations(image=image_ocv_right)
            masked_image_right = hand_right.get_hand_mask(gmm_model=load_model)

            cv2.imshow("Image", masked_image_left)
                   
            gray = cv2.hconcat([masked_image_left,masked_image_right])
            gray = np.uint8(gray)
            masked_out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            
            # gray_1 = cv2.cvtColor(image_ocv_left, cv2.COLOR_BGR2GRAY)
            # gray_2 = cv2.cvtColor(image_ocv_right, cv2.COLOR_BGR2GRAY)        
            # gray = cv2.hconcat([gray_1,gray_2])
            image_ocv_left = image_ocv_left[:,:,0:3]
            image_ocv_right = image_ocv_right[:,:,0:3]
            regular_left_out.write(image_ocv_left)
            regular_right_out.write(image_ocv_right)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

    
    masked_out.release()
    regular_left_out.release()
    regular_right_out.release()
    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    # cal = Calibration()
    # cal.capture_hand()
    # cal.gmm_train()
    import pickle
    
    file_name = 'hand_gmm_model.sav'
    load_model = pickle.load(open( file_name, "rb" ))
    main()