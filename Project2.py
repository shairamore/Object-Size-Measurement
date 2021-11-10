import numpy as np
import cv2

img_path = "example_01.jpg"
image = cv2.imread(img_path)
cv2.imshow('',image)
cv2.waitKey(0) 


grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grayscale, (9, 9), 0)
img_edge = cv2.Canny(blur, 50, 100)
img_edge = cv2.dilate(img_edge, None, iterations=1)
img_edge = cv2.erode(img_edge, None, iterations=1)

cv2.imshow('',img_edge)
cv2.waitKey(0) 

