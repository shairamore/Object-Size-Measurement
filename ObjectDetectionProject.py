#importing the libraries
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


videoCaptureObject = cv2.VideoCapture(0)  #here 1 represents secondary camera
result = True

# set the brightness of image captured by the camera
videoCaptureObject.set(10,160)
videoCaptureObject.set(3,1920)
videoCaptureObject.set(4,1080)

while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite("NewPicture.jpg",frame)
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()

img_path = "NewPicture.jpg" #image path for dynamically clicked picture
#img_path = "example_01.jpg"  #image path for example image(static) 
image = cv2.imread(img_path)
cv2.imshow('IMAGE ',image)  #syntax of imshow - imshow('name of the window',image)
cv2.waitKey(0)


#Preprocessing the image
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #converting the color to grayscale
blur = cv2.GaussianBlur(grayscale, (9, 9), 0)
img_edge = cv2.Canny(blur, 50, 100)
img_edge = cv2.dilate(img_edge, None, iterations=1)
img_edge = cv2.erode(img_edge, None, iterations=1)

#cv2.imshow('  ',img_edge)
#cv2.waitKey(0) 


#Object Segmentation
# Finding Contours
cnts = cv2.findContours(img_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)  

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

#cv2.drawContours(img_edge, cnts, -1, (0,255,0), 3)

#cv2.imshow(" ", img_edge)
#cv2.waitKey(0) 

#Creating reference box
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box #tl = top left ; tr= top right ; br= bottom right ; bl=bottom left
dist_in_pixel = euclidean(tl, tr)  #finding euclidean distance between the opposite sides of the box
dist_in_cm = 2
pixel_per_cm = dist_in_pixel/dist_in_cm

#repeating the process for every object contour in the image
for cnt in cnts:
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	cv2.drawContours(image, [box.astype("int")], -1, (255, 0, 0), 2)
	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
	wid = euclidean(tl, tr)/pixel_per_cm
	ht = euclidean(tr, br)/pixel_per_cm
	cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow(' ',image)
cv2.waitKey(0) 
