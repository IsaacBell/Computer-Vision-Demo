import cv2, time, pandas
from datetime import datetime

# img = cv2.imread("C:/Users/I Am/Documents/RHT/RTG/images/expeditor.png",1) #color, default
# img_1 = cv2.imread("C:/Users/I Am/Documents/RHT/RTG/images/expeditor.png",0) #b-w/grayscale
# # img = cv2.imread("C:/Users/I Am/Pictures/Dr. Bergana.jpeg",1) #color, default
# # # img_1 = cv2.imread("C:/Users/I Am/Pictures/Dr. Bergana.jpeg",0) #b-w/grayscale
# # #resized = cv2.resize(img_1,(600,600))
# # resized = cv2.resize(img_1, (int(img_1.shape[1]*2),int(img_1.shape[0]*2)))
# # cv2.imshow("Legend",resized)
# # cv2.waitKey(0)

# # cv2.destroyAllWindows()

# #CascadeClassifier object!

# img = cv2.imread("C:/Users/I Am/Pictures/Dr. Bergana.jpeg")
# # img = cv2.imread("C:/Users/I Am/Pictures/photo.jpg")

# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# resized = cv2.resize(img, (int(img.shape[1]*2),int(img.shape[0]*2)))
# cv2.imshow("Fail detection",resized)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# #
# a = 1
first_frame = None
statoos = [None,None]
times = []
df = pandas.DataFrame(columns=["Start","End"])
face_cascade = cv2.CascadeClassifier("C:/Users/I Am/Documents/CVjob/Camera-Capture-master/Camera-Capture-master/opencv-4.4.0/data/haarcascades/haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier("C:/Users/glennvolkerding/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0/LocalCache/local-packages/Python38/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
	# a = a + 1
	check, frame = video.read()
	print (frame)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(21,21),cv2.COLOR_BGR2GRAY)
	if first_frame is None: #set first_frame to NOne every time motion detector is reactivated.  Allow for timer feature
		first_frame = gray
		continue
	delta_frame = cv2.absdiff(first_frame,gray)
	thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
	thresh_delta = cv2.dilate(thresh_delta,None, iterations = 0 )
	(cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in cnts:
		if cv2.contourArea(contour) < 1000: continue
		statoos = 1
		( x,y,w,h ) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)

	cv2.imshow('frame', frame)
	cv2.imshow('Here you is, press Q to quit', gray)
	cv2.imshow('delta', delta_frame)
	cv2.imshow('thresh', thresh_delta)


	key = cv2.waitKey(1)
	if key == ord('q'):
	    break

video.release()
