import numpy as np
import cv2 as cv

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
	img = cv.imread('datos-T2/yosemite/Yosemite1.jpg')
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

	surf = cv.xfeatures2d.SURF_create(100, 4, 3, 0, 0)
	kp, descriptor = surf.detectAndCompute(gray,None)

	kp_octaves = agruparOctavas(kp)

	print("Numero de keypoints: ", len(kp))

	for octave, points in kp_octaves.items():
		img=cv.drawKeypoints(img,points,img,color(octave))
		

	cv.imwrite('surf_keypoints.jpg',img)


if __name__ == '__main__':
	main()
