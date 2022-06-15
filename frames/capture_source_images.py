# program to capture single image from webcam in python

# importing OpenCV library
#from cv2 import *

import cv2

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cam_port_left = 2
cam_left = cv2.VideoCapture(cam_port_left)

cam_port_right = 3
cam_right = cv2.VideoCapture(cam_port_right)

cv2.waitKey(0)

for i in range(10):
	# reading the input using the camera
	result, image_left = cam_left.read()
	result, image_right = cam_right.read()

	# showing result, it take frame name and image
	# output
	cv2.imshow("ImgLeft", image_left)
	cv2.imshow("ImgRight", image_right)

	# saving image in local storage
	cv2.imwrite("source_images/left/ImgLeft" + str(i) + ".png", image_left)
	cv2.imwrite("source_images/right/ImgRight" + str(i) + ".png", image_right)

	# If keyboard interrupt occurs, destroy image
	# window
	cv2.waitKey(0)
	cv2.destroyWindow("ImgLeft")
	cv2.destroyWindow("ImgRight")
