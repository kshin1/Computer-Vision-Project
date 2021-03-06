# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import pickle
import os

happy = {}
sad = {}
angry = {}
neutral = {}
surprised = {}
confused = {}
other = {}

EXPRESSION_DICT = {"h": happy, "sa": sad, "a": angry,\
		"n": neutral, "su": surprised, "c": confused, "o": other}
EXPRESSION = {"h":"happy", "sa": "sad", "a": "angry",\
		"n": "neutral", "su": "surprised", "c": "confused", "o": "other"}
EXPRESSION_TEST_DICT = {"h": happy, "sa": sad, "a": angry,\
		"n": neutral, "su": surprised, "c": confused, "o": other}

for key, d in EXPRESSION_DICT.items():
	if os.path.isfile(EXPRESSION[key] + ".pickle"):
		f = open(EXPRESSION[key] + ".pickle", "rb")
		EXPRESSION_DICT[key] = pickle.load(f)
		f.close()

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# load the input image, resize it, and convert it to grayscale
#for img in glob.glob("../projectData/*.bmp"): #THIS IS FOR ALL IMAEGS
print("Enter 'quit' to stop tagging images")
for img in glob.glob("projectImages/*.bmp"): #THIS IS FOR TRAINING DATA
#for img in glob.glob("projectImages/A[0-9]*.bmp"): # Try on test subject 1
	img_id = img.split("/")[1]
	# Check if img has already been tagged
	tagged = False
	for key, d in EXPRESSION_DICT.items():
		# Already tagged the image
		if img_id in d.keys():
			tagged = True
			break
	# Go to the next image
	if tagged:
		continue

	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	# show the output image with the face detections + facial landmarks
	#cv2.imshow("Output", image)

	#cv2.waitKey(0)
	
	facial_features = {}
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		facial_features[name] = shape[i:j]
	
	# User tagging facial expression
	expression = raw_input("{}: ".format(img_id))
	while expression not in EXPRESSION_DICT.keys() and expression != "quit":
		expression = raw_input("{}: ".format(list(EXPRESSION_DICT.keys())))
	
	# Stop tagging images
	if expression == "quit":
		break

	EXPRESSION_DICT[expression][img_id] = facial_features
	
# Save classifications to JSON files
for e, d in EXPRESSION_DICT.items():
	f = open(EXPRESSION[e] + ".pickle", "wb")
	pickle.dump(d, f)
	f.close()


'''for img in glob.glob("projectImages/[K-M][0-9]*.bmp"): #THIS IS FOR TEST DATA
	img_id = img.split("/")[1]
	# Check if img has already been tagged
	tagged = False
	for key, d in EXPRESSION_DICT.items():
		# Already tagged the image
		if img_id in d.keys():
			tagged = True
			print key
			break
	# Go to the next image
	if tagged:
		continue

	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)

	cv2.waitKey(0)
	
	facial_features = {}
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		facial_features[name] = shape[i:j]
	
	# User tagging facial expression
	expression = raw_input("{}: ".format(img_id))
	while expression not in EXPRESSION_DICT.keys() and expression != "quit":
		expression = raw_input("{}: ".format(list(EXPRESSION_DICT.keys())))
	
	# Stop tagging images
	if expression == "quit":
		break

	EXPRESSION_TEST_DICT[expression][img_id] = facial_features
'''
