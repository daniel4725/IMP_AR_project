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
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

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
            
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv_left = image_zed_left.get_data()
            image_ocv_right = image_zed_right.get_data()
            
            img = image_ocv_left[:, 0:int(image_ocv_left.shape[1] / 2), :]  # left image
            hand = HandOperations(image=img)
            masked_image = hand.get_hand_mask(gmm_model=cal.GMM_Model)
            depth_image_ocv = depth_image_zed.get_data()

            cv2.imshow("Image", masked_image)
            # cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    cal = Calibration()
    cal.capture_hand()
    cal.gmm_train()
    
    main()