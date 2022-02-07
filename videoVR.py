from aux_functions import *

left_vid = cv2.VideoCapture('hands/left.mp4')
right_vid = cv2.VideoCapture('hands/right.mp4')


while True:
    # Capture frame-by-frame
    ret1, im_l = left_vid.read()
    ret2, im_r = right_vid.read()
    if not (ret1 and ret2):
        left_vid = cv2.VideoCapture('hands/left.mp4')
        right_vid = cv2.VideoCapture('hands/right.mp4')
        continue
    factor = 1.4
    dim = (int(im_l.shape[1]*factor*0.6), int(im_l.shape[0]*factor))
    factor = 0.6
    dim = (int(im_l.shape[1]*factor), int(im_l.shape[0]*factor))
    im_l = cv2.resize(im_l, dim, interpolation=cv2.INTER_AREA)
    im_r = cv2.resize(im_r, dim, interpolation=cv2.INTER_AREA)
    distorter = ImageDistorter(im_l)
    out = distorter.distort_and_concat(im_l, im_r)
    cv2.imshow('left', out)
    cv2.imshow('right', im_r)
    if cv2.waitKey(80) & 0xFF == 27:  # press 'Esc' to exit
      break
left_vid.release()
right_vid.release()
cv2.destroyAllWindows()