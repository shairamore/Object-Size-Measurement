#importing the libraries
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

#storing the image in a variable
img_path = "example_01.jpg"
image = cv2.imread(img_path)
cv2.imshow('Example 1 Image',image)  #syntax of imshow - imshow('name of the window',image)
cv2.waitKey(0)   #holding the window with the image on the screen till user presses any button


#Preprocessing the image
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #converting the color to grayscale
blur = cv2.GaussianBlur(grayscale, (9, 9), 0)
img_edge = cv2.Canny(blur, 50, 100)
img_edge = cv2.dilate(img_edge, None, iterations=1)
img_edge = cv2.erode(img_edge, None, iterations=1)

cv2.imshow('Preprocessed Image - Example 1',img_edge)
cv2.waitKey(0) 


#Object Segmentation
# Finding Contours
cnts = cv2.findContours(img_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = contours.sort_contours(cnts)  

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

cv2.drawContours(img_edge, cnts, -1, (0,255,0), 3)

cv2.imshow(" ", img_edge)
cv2.waitKey(0) 
