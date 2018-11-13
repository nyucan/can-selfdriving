import cv2
import numpy as np

#For the color filter
lower_red1 = np.array([0,160,50])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([175,160,50])
upper_red2 = np.array([180,255,255])

class processImage():
	
	def redFilter(self, im): 
		imgRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		imgHsv = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
		redIMG = cv2.inRange(imgHsv, lower_red1, upper_red1) + cv2.inRange(imgHsv, lower_red2, upper_red2)
		return redIMG
	def getRectangle(self, contours):
		areas = [cv2.contourArea(c) for c in contours]
		max_ind = np.argmax(areas)
		cnt = contours[max_ind]
		x,y,w,h = cv2.boundingRect(cnt)
		return x,y,w,h

	def drawLine(self, im, rho, theta):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                return cv2.line(im, (x1,y1), (x2,y2), (255,0,255), 2)

        def getLines(self, im):
                im = cv2.resize(im, (0,0), fx=0.3, fy=4)
                imgTrim = im[int(im.shape[0]*0.90):,:]
                edges = cv2.Canny(imgTrim, 110, 160)
                #kernelDilate = np.ones((2,2), np.uint8)
                #kernelErode = np.ones((3,3), np.uint8)
                #imgDilate = cv2.dilate(edges, kernelDilate, iterations = 3)
                #imgErode = cv2.erode(imgDilate, kernelErode, iterations = 1)
                lines = cv2.HoughLines(edges, 1, np.pi/360, 100)
                return lines, imgTrim

        def drawCircle(self, im, x, y, radius, color):
                return cv2.circle(im, (x, y), radius, color, -1)

        def calculateEdgePoint(self, rho, theta, y):
                a = np.cos(theta)
                b = np.sin(theta)
                return (rho-(y*b))/a
