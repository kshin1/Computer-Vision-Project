import numpy as py
import cv2
import os
import sys

DIR = '.'
EXTENSIONS = {'png', 'jpg'}

pictures = []

# Grab all images in given directory
for item in os.listdir(DIR):
	extension = item.split('.')[-1]
	if extension in EXTENSIONS:
		pictures.append(item)

# Display the pictures database
print "Pictures in database:"
for pic in pictures:
	print pic

# Get user image
my_pic = raw_input("Select a picture to display: ")
while my_pic not in pictures:
	my_pic = raw_input("Invalid image: select a picture to display: ")

# Load a color image
img = cv2.imread(my_pic)

# Display image - 'image' is the window name
cv2.imshow('image',img)
# Keyboard binding function (arg is time in ms)
	# arg 0 means that ANY key pressed will cause the program to continue
k = cv2.waitKey(0)

if k == 27: # wait for ESC key to exit
	# Destroy all windows created
		# single arg passed in will close that specific window
	cv2.destroyAllWindows()
