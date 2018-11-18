import cv2 as cv
import numpy as np

def canvas(central_img, size):
	canvas = np.copy(central_img).astype(np.uint8)

	nrow_extra = size[0] - central_img.shape[0]
	ncol_extra = size[1] - central_img.shape[1]

	row = np.zeros((1,canvas.shape[1],3))

	for i in range(nrow_extra // 2):
		canvas = np.vstack((row, canvas))
		canvas = np.vstack((canvas, row))

	col = np.zeros((canvas.shape[0],1,3))

	for i in range(ncol_extra // 2):
		canvas = np.hstack((col, canvas))
		canvas = np.hstack((canvas, col))


	return canvas

def main():
	img1 = cv.imread("datos-T2/mosaico-1/mosaico002.jpg")
	img2 = cv.imread("datos-T2/mosaico-1/mosaico003.jpg")
	img3 = cv.imread("datos-T2/mosaico-1/mosaico004.jpg")

	gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
	gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
	gray_img3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

	#Ajustamos Parametros SIFT para la imagen mosaico00X.jpg
	#EdgeThreshold es una cota inferior. Es la minima "curvatura" que debe tener una arista para ser considerada
	edgeThreshold = 5

	#EdgeContrast es una cota inferior. Es el minimo contraste que debe de haber entre un punto y su entorno para tenerlo en cuenta
	contrastThreshold = 0.03

	#Creamos un detector SHIFT con los parametros anteriores 
	sift1 = cv.xfeatures2d.SIFT_create(contrastThreshold = contrastThreshold, edgeThreshold = edgeThreshold)

	#Detectamos los puntos y calculamos sus respectivo descripores sobre mosaico002.jpg en escala de grises
	keypoints1, descriptors1 = sift1.detectAndCompute(gray_img1,None)
	
	#Pintamos los keyPoints de la imagen sobre ella
	img_kp1=cv.drawKeypoints(img1,keypoints1,None,(255,0,0))


	#Creamos un detector SHIFT con los parametros anteriores 
	sift2 = cv.xfeatures2d.SIFT_create(contrastThreshold = contrastThreshold, edgeThreshold = edgeThreshold)

	#Detectamos los puntos y calculamos sus respectivo descripores sobre mosaico003.jpg en escala de grises
	keypoints2, descriptors2 = sift2.detectAndCompute(gray_img2,None)

	#Pintamos los keyPoints de la imagen sobre ella
	img_kp2 = cv.drawKeypoints(img2, keypoints2, None, (255,0,0))


	#Creamos un detector SHIFT con los parametros anteriores 
	sift3 = cv.xfeatures2d.SIFT_create(contrastThreshold = contrastThreshold, edgeThreshold = edgeThreshold)

	#Detectamos los puntos y calculamos sus respectivo descripores sobre mosaico004.jpg en escala de grises
	keypoints3, descriptors3 = sift3.detectAndCompute(gray_img3,None)

	#Pintamos los keyPoints de la imagen sobre ella
	img_kp3 = cv.drawKeypoints(img3, keypoints3, None, (255,0,0))

	#Definimos el 'matcher' (BruteForce+crossCheck)
	matcher = cv.BFMatcher_create(crossCheck = 1)

	#DrawN -> Nº de matches que dibujar
	drawN = 100

	#Emparejamos los keypoint de ambas imagenes
	matches12 = matcher.match(descriptors1, descriptors2)

	#Dibujamos los matches entre ambas imagenes
	img_matching12 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches12[:drawN], None)

	#Emparejamos los keypoint de ambas imagenes
	matches23 = matcher.match(descriptors2, descriptors3)

	#Dibujamos los matches entre ambas imagenes
	img_matching23 = cv.drawMatches(img2, keypoints2, img3, keypoints3, matches23[:drawN], None)

	#Transformamos los keypoints en puntos de las imagenes Mosaico002-3.jpg
	src_points1 = np.float32([ keypoints1[m.queryIdx].pt for m in matches12 ]).reshape(-1,1,2)
	dst_points2 = np.float32([ keypoints2[m.trainIdx].pt for m in matches12 ]).reshape(-1,1,2)

	#Hallamos la homografia entre Mosaico002-3.jpg
	homography12, mask12 = cv.findHomography(src_points1, dst_points2, cv.RANSAC, 1)
	
	#Calculamos la transformacion de perspectiva de  Mosaico002.jpg en Mosaico003.jpg
	h,w,d = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	perspTrans12 = cv.perspectiveTransform(pts,homography12)

	#Dibujamos la transformacion de perspectiva de  Mosaico002.jpg en Mosaico003.jpg
	img_perspTrans12 = cv.polylines(np.copy(img2),[np.int32(perspTrans12)],True,255,3, cv.LINE_AA)

	#Transformamos la imagen Mosaico002.jpg
	img_perspTrans1 = cv.warpPerspective(img1, homography12, (img1.shape[1], img1.shape[0]))

	#######################################################################################################

	#Transformamos los keypoints en puntos de las imagenes Mosaico002-3.jpg
	src_points3 = np.float32([ keypoints3[m.trainIdx].pt for m in matches23 ]).reshape(-1,1,2)
	dst_points2 = np.float32([ keypoints2[m.queryIdx].pt for m in matches23 ]).reshape(-1,1,2)

	#Hallamos la homografia entre Mosaico002-3.jpg
	homography23, mask23 = cv.findHomography(src_points3, dst_points2, cv.RANSAC, 1)
	
	#Calculamos la transformacion de perspectiva de  Mosaico002.jpg en Mosaico003.jpg
	h,w,d = img3.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	perspTrans23 = cv.perspectiveTransform(pts,homography23)

	#Dibujamos la transformacion de perspectiva de  Mosaico002.jpg en Mosaico003.jpg
	img_perspTrans23 = cv.polylines(np.copy(img2),[np.int32(perspTrans23)],True,255,3, cv.LINE_AA)

	#Transformamos la imagen Mosaico002.jpg
	img_perspTrans3 = cv.warpPerspective(img3, homography23, (img3.shape[1], img3.shape[0]))


	cv.imshow("Mosaico002.jpg", img1)
	cv.imshow("Mosaico003.jpg", img2)

	print("Nº KeyPoints Mosaico002.jpg -> ", len(keypoints1))
	#cv.imshow("KeyPoints Mosaico002.jpg", img_kp1)

	print("Nº KeyPoints Mosaico003.jpg -> ", len(keypoints2))
	#cv.imshow("KeyPoints Mosaico003.jpg", img_kp2)

	#print("Nº KeyPoints Mosaico004.jpg -> ", len(keypoints3))
	#cv.imshow("KeyPoints Mosaico004.jpg", img_kp3)

	#cv.imshow("Match Mosaico002-3.jpg", img_matching12)

	#cv.imshow("Match Mosaico003-4.jpg", img_matching23)


	cv.imshow("Perspective Mosaico002.jpg", img_perspTrans12)
	cv.imshow("Transformed Mosaico002.jpg", img_perspTrans1)

	cv.imshow("Perspective Mosaico003.jpg", img_perspTrans23)
	#cv.imshow("Transformed Mosaico003.jpg", img_perspTrans3)

	cv.imshow("Transformed Mosaico003.jpg", canvas(img2, img2.shape*2))




	cv.waitKey(0)
	cv.destroyAllWindows()
if __name__ == '__main__':
	main()
