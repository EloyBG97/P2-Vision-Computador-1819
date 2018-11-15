import numpy as np
import cv2 as cv

img = cv.imread('datos-T2/yosemite/Yosemite1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.4)

kp = sift.detect(gray,None)

print("Numero de keypoints: ", len(kp))

img = cv.drawKeypoints(img,kp,img)

cv.imwrite('sift_keypoints.jpg',img)
