import numpy as np
import cv2 as cv
from random import randrange



def agruparOctavas(keyPoints):
	kp_octaves = {}

	for kp in keyPoints:
		if not str(kp.octave) in kp_octaves:
			kp_octaves[str(kp.octave)] = []

		kp_octaves[str(kp.octave)].append(kp)

	return kp_octaves

def color(octave):
	base = int(octave) % (256*256*256)


	r = base % 256
	g = (base // 256) % 256
	b = (base // (256*256)) % 256

	return (b,g,r)



def main():
	img1 = cv.imread('datos-T2/yosemite/Yosemite1.jpg')
	gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

	img2 = cv.imread('datos-T2/yosemite/Yosemite2.jpg')
	gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

	sift1 = cv.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)
	kp1, descriptor1 = sift1.detectAndCompute(gray1,None)
	

	sift2 = cv.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)
	kp2, descriptor2 = sift2.detectAndCompute(gray2,None)

	matcher = cv.BFMatcher_create()
	matches = matcher.knnMatch(descriptor1, descriptor2,2)

	good = []

	for m,n in matches:
       		good.append(m)
	

	res = cv.drawMatches(img1, kp1, img2, kp2, good[:100], None)

	cv.imwrite("correspondencias_yosemite.jpg", res)


if __name__ == '__main__':
	main()
