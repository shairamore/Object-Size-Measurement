#importing the libraries
import numpy as np
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

