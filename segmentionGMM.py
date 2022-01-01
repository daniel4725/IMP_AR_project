import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.mixture import GaussianMixture
import time

image = cv2.imread('handTest2.jpeg')
originalShape = image.shape

imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

L = np.array(imageLAB[:, :, 0]).flatten()
a = np.array(imageLAB[:, :, 1]).flatten()
b = np.array(imageLAB[:, :, 2]).flatten()

R = np.array(image[:, :, 2]).flatten()
G = np.array(image[:, :, 1]).flatten()
B = np.array(image[:, :, 0]).flatten()

# vrow = np.array([range(originalShape[0])])
# x = np.array(np.vstack([vrow] * originalShape[1])).flatten()
#
# vcol = np.array([range(originalShape[1])]).transpose()
# y = np.array(np.hstack([vcol] * originalShape[0])).flatten()

# data = np.array([R, G, B, x, y]).transpose()
# data = np.array([R, G, B]).transpose()
data = np.array([a, b]).transpose()

n_compononts = 2
classif = GaussianMixture(n_components=n_compononts)
classif.fit(data)
GMM_Labels = classif.predict(data)

import pickle   # pip install pickle-mixin

# save the model to disk
filename = 'hand_gmm_model.sav'
pickle.dump(classif, open(filename, 'wb'))

segmented = np.array(GMM_Labels).reshape(originalShape[0], originalShape[1])

weights = classif.weights_
print(weights)
means = classif.means_
print(means)
variance = classif.covariances_
print(variance)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(segmented)
ax.set_title(f"Segmented with {n_compononts} components")
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax.set_title('Original')
plt.show()



# handMean = means[1]
# handVariance = variance[1]
# #
# image2 = cv2.imread('triangle.jpg')
# image2Shape = image2.shape
# image2LAB = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

# L2 = np.array(image2LAB[:, :, 0]).flatten()
# a2 = np.array(image2LAB[:, :, 1]).flatten()
# b2 = np.array(image2LAB[:, :, 2]).flatten()

# data2 = np.array([a2, b2]).transpose()
# GMM_Labels2 = classif.predict(data2)

# segmented2 = np.array(GMM_Labels2).reshape(image2Shape[0], image2Shape[1])

# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# imgplot = plt.imshow(segmented2)
# ax.set_title(f"Segmented with {n_compononts} components")
# ax = fig.add_subplot(1, 2, 2)
# imgplot = plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
# ax.set_title('Original')
# plt.show()

# cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture('http://192.168.0.131:4747/mjpegfeed?640x480')
cap = cv2.VideoCapture('shapes.mp4')
out = cv2.VideoWriter('gmm_shapes.mp4', -1, 20.0, (640,480))
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# cap.set(CV_CAP_PROP_BUFFERSIZE, 3)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


while True:
    try:
        suc, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # start time to calculate FPS
        start = time.time()

        # flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        prevgray = gray

        # End time
        
        # calculate the FPS for current frame detection
        

        # cv2.imshow('flow', draw_flow(gray, flow))
        
        image2 = img
        image2Shape = image2.shape
        image2LAB = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        L2 = np.array(image2LAB[:, :, 0]).flatten()
        a2 = np.array(image2LAB[:, :, 1]).flatten()
        b2 = np.array(image2LAB[:, :, 2]).flatten()

        data2 = np.array([a2, b2]).transpose()
        GMM_Labels2 = classif.predict(data2)

        segmented2 = np.array(GMM_Labels2).reshape(image2Shape[0], image2Shape[1])
        
        segmented2_norm = cv2.normalize(segmented2, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        end = time.time()
        fps = 1 / (end - start)
        print(f"{fps:.2f} FPS")

        cv2.imshow('flow', segmented2_norm)
        out.write(segmented2_norm)
        # cv2.imshow('flow HSV', draw_hsv(flow))
        # cv2.imshow('contour', draw_contour_masked(gray, flow))

    except:
        pass
    
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    


cap.release()
out.release()
cv2.destroyAllWindows()