import numpy as np
import cv2 as cv

img = cv.imread('datos-T2/yosemite/Yosemite1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

surf = cv.xfeatures2d.SURF_create(100, 4, 3, 0, 0)

kp = surf.detect(gray,None)

print("Numero de keypoints: ", len(kp))

img = cv.drawKeypoints(img,kp,img)

cv.imwrite('surf_keypoints.jpg',img)
