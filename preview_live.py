import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.mixture import GaussianMixture
import time
import pickle   # pip install pickle-mixin
import keras

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# load the model from disk
filename = 'hand_gmm_model.sav'
gmm_model = pickle.load(open(filename, 'rb'))

filename = 'ML\keras_fingers_model'
keras_model = keras.models.load_model(filename)
print(keras_model.summary())


img_size = 128
cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture('http://192.168.0.169:4747/mjpegfeed?640x480')

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# cap.set(CV_CAP_PROP_BUFFERSIZE, 3)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


while True:

    suc, img = cap.read()
    
    image2 = img
    image2Shape = image2.shape
    image2LAB = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

    L2 = np.array(image2LAB[:, :, 0]).flatten()
    a2 = np.array(image2LAB[:, :, 1]).flatten()
    b2 = np.array(image2LAB[:, :, 2]).flatten()

    data2 = np.array([a2, b2]).transpose()
    GMM_Labels2 = gmm_model.predict(data2)
    
    segmented2 = np.array(GMM_Labels2).reshape(image2Shape[0], image2Shape[1])
    
    segmented2_norm = cv2.normalize(segmented2, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
    gray_smaller = cv2.resize(segmented2_norm, (img_size,img_size)).flatten()
        
    gray_smaller_1D = np.reshape(gray_smaller,(1,img_size*img_size))
    
    gray_smaller_2D = np.reshape(gray_smaller,(1,img_size,img_size))
    
    # y_pred=loaded_model.predict(gray_smaller_1D) # sklearn model
    y_pred_array = keras_model.predict(gray_smaller_2D)
    y_pred = np.argmax(y_pred_array, axis=1)
    
    predict = str(y_pred)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(segmented2_norm, predict, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    
    # gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.imshow('flow', segmented2_norm)
    

    key = cv2.waitKey(5)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()