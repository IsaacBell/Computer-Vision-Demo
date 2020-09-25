import cv2, time, pandas
# import os, sys
#from cv2 import cv2 #ended IDE warnings
from datetime import datetime
import pandas


def nothing(x):
    pass

period = 0
threshold = 30
blur = 21
scale_factor = 1.05

def timeChange(x):
	global period
	period = x
	# period = cv2.getTrackbarPos('time',main_title)

def blurChange(x):
	global blur
	blur = int(round(x))
	# blur = int(round(cv2.getTrackbarPos('blur',gray_title)))

def scaleChange(x):
	global scale_factor
	scale_factor = (201+cv2.getTrackbarPos('scale',main_title)) / 200

def threshChange(x):
	global threshold
	threshold = x

activated = False

# #
a = 1
first_frame = None
statoos = [None,None]
times = []
df = pandas.DataFrame(columns=["Start","End"])
face_cascade = cv2.CascadeClassifier("datasets/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# video = cv2.VideoCapture("C:/Users/I Am/Videos/more castlevania/Carmilla in the Catacombs.mp4")#,cv2.CAP_V4L2)#, cv2.CAP_DSHOW)
face_finding = True
motion_detecting = False
#threshVal1 = 30
main_title = 'Press F to toggle face detection, M to toggle motion detection'
gray_title = 'Here u is, press Q to quit'
thresh_title = 'Press R to reset frame'
cv2.namedWindow(main_title)
if face_finding:
	cv2.createTrackbar('scale',main_title,9,79,scaleChange)
	# cv2.setTrackbarPos('scale',main_title, 105)
	cv2.createTrackbar('min',main_title,4,8,nothing)
cv2.namedWindow(gray_title)
if motion_detecting:
   cv2.createTrackbar('blur',gray_title,21,29,blurChange)
   cv2.namedWindow('delta')
   cv2.createTrackbar('time','delta',0,255,timeChange)
   cv2.namedWindow(thresh_title)
   cv2.createTrackbar('thresh',thresh_title,30,255,nothing)
# cv2.setTrackbarPos('blur',gray_title, 21)
last_time = time.time()
errors = 0
while True:
	check, frame = video.read()
	#print (frame)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if motion_detecting:
		try:
			grayB = cv2.GaussianBlur(gray,(blur,blur),cv2.COLOR_BGR2GRAY)
		except:
			grayB = cv2.GaussianBlur(gray,(21,21),cv2.COLOR_BGR2GRAY)
		if period != 0:
			now = time.time()
			if (now - last_time) > period:
				last_time = now
				first_frame = None
		if first_frame is None: #set first_frame to NOne every time motion detector is reactivated.  Allow for timer feature
			first_frame = grayB
			continue
		delta_frame = cv2.absdiff(first_frame,grayB)
		thresh_delta = cv2.threshold(delta_frame,threshold,255,cv2.THRESH_BINARY)[1]
		thresh_delta = cv2.dilate(thresh_delta,None, iterations = 0 )
		(cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for contour in cnts:
			if cv2.contourArea(contour) < 1000: continue
			statoos = 1
			( x,y,w,h ) = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),3)
	if face_finding:
		min_neighbors = 1 + cv2.getTrackbarPos('min',main_title)
		faces = face_cascade.detectMultiScale(gray, scaleFactor = scale_factor, #smaller scaleFactor = "greater accuracy" (precision?) 
														minNeighbors=min_neighbors)
		# print(type(faces))
		# print(faces)
		for x,y,w,h in faces: #rectangle method
			frame = cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3) #rectangle(image, (line origin), (line endpoint), rgb of outline, width)
			if motion_detecting: grayB = cv2.rectangle(grayB, (x,y), (x+w,y+h),(0,255,0),3) 
			else: gray = cv2.rectangle(gray, (x,y), (x+w,y+h),(0,255,0),3)#repeat for gray

	cv2.imshow(main_title, frame)
	cv2.imshow(gray_title, (grayB if motion_detecting else gray))
	if motion_detecting:
		cv2.imshow('delta', delta_frame)
		cv2.imshow(thresh_title, thresh_delta)

	key = cv2.waitKey(1)
	if key == ord('q'):
	    break
	if key == ord('r'): first_frame = None
	if key == ord('f'):
		cv2.destroyWindow(main_title)
		cv2.namedWindow(main_title)
		if not face_finding:
			cv2.createTrackbar('scale',main_title,9,79,scaleChange)
			cv2.createTrackbar('min',main_title,4,8,nothing)
		face_finding = not face_finding
	if key == ord('m'):
		if motion_detecting:
		   motion_detecting = False
		   cv2.destroyWindow(thresh_title)
		   cv2.destroyWindow('delta')
		   cv2.destroyWindow(gray_title)
		   cv2.namedWindow(gray_title)
		else:
			cv2.namedWindow(thresh_title)
			cv2.namedWindow('delta')
			motion_detecting = True
			cv2.destroyWindow(gray_title)
			cv2.namedWindow(gray_title)
			cv2.createTrackbar('blur',gray_title,21,29,blurChange)
			cv2.namedWindow('delta')
			cv2.createTrackbar('time','delta',0,255,timeChange)
			cv2.namedWindow(thresh_title)
			cv2.createTrackbar('thresh',thresh_title,30,255,threshChange)
			cv2.destroyWindow(main_title)
			cv2.namedWindow(main_title)
#video.release() #froze it: why?
cv2.destroyAllWindows()
quit() #still won't terminate