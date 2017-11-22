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
import math

happy = {}
sad = {}
angry = {}
neutral = {}
surprised = {}
confused = {}
other = {}
countH = 0
countSA = 0
countA = 0
countN = 0
countSU = 0
countC = 0
countO = 0
hList = [0,0,0,0,0,0]
saList = [0,0,0,0,0,0]
aList = [0,0,0,0,0,0]
nList = [0,0,0,0,0,0]
suList = [0,0,0,0,0,0]
cList = [0,0,0,0,0,0]
oList = [0,0,0,0,0,0]

EXPRESSION_DICT = {"h": happy, "sa": sad, "a": angry,\
		"n": neutral, "su": surprised, "c": confused, "o": other}
EXPRESSION = {"h":"happy", "sa": "sad", "a": "angry",\
		"n": "neutral", "su": "surprised", "c": "confused", "o": "other"}
AVE_EXPRESSION_DICT = {"h": [], "sa": [], "a": [],\
		"n": [], "su": [], "c": [], "o": []}
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
for img in glob.glob("projectImages/[A-J][0-9]*.bmp"): #THIS IS FOR TRAINING DATA
#for img in glob.glob("projectImages/A[0-9]*.bmp"): # Try on test subject 1
	img_id = img.split("/")[1]
	# Check if img has already been tagged
	tagged = False
	facial_features = {}
	for key, d in EXPRESSION_DICT.items():
		# Already tagged the image
		if img_id in d.keys():
			tagged = True
			facial_features = d[img_id]
			break
	# Go to the next image if it has not been tagged
	if not tagged:
		continue
	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	#print(facial_features)
	
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

		# Process the person's left eye (our right)
		lEye_x = []
		lEye_y = []
		left_most_x = facial_features["left_eye"][0][0]
		left_most_y = facial_features["left_eye"][0][1]
		top_most_x = facial_features["left_eye"][0][0]
		top_most_y = facial_features["left_eye"][0][1]
                for (x, y) in facial_features["left_eye"]:
			if x < left_most_x: 
				left_most_x = x
				left_most_y = y
			if y <= top_most_y:
				top_most_x = x
				top_most_y = y
		lEye_x = [top_most_x, left_most_x]
		lEye_y = [top_most_y, left_most_y]		

		cv2.circle(image, (top_most_x, top_most_y), 1, (0, 255, 0), -1)
		cv2.circle(image, (left_most_x, left_most_y), 1, (0,255,0), -1)

		# Process the person's right eye (our left)
		rEye_x = []
		rEye_y = []
		top_most_x = facial_features["right_eye"][0][0]
		top_most_y = facial_features["right_eye"][0][1]
		right_most_x = facial_features["right_eye"][0][0]
		right_most_y = facial_features["right_eye"][0][1]
                for (x, y) in facial_features["right_eye"]:
			if y <= top_most_y:
				top_most_x = x
				top_most_y = y
			if x > right_most_x:
				right_most_x = x
				right_most_y = y
		rEye_x = [top_most_x, right_most_x]
		rEye_y = [top_most_y, right_most_y]

		cv2.circle(image, (right_most_x, right_most_y), 1, (0, 255, 0), -1)
		cv2.circle(image, (top_most_x, top_most_y), 1, (0,255,0), -1)


		# Process the mouth
		left_most_x = facial_features["mouth"][0][0]
		left_most_y = facial_features["mouth"][0][1]
		right_most_x = facial_features["mouth"][0][0]
		right_most_y = facial_features["mouth"][0][1]
		top_y = facial_features["mouth"][0][1]
		bottom_y = facial_features["mouth"][0][1]
		for (x, y) in facial_features["mouth"]:
			if x < left_most_x: 
				left_most_x = x
				left_most_y = y
			if x > right_most_x:
				right_most_x = x
				right_most_y = y
			if y < top_y:
				top_y = y
			if y > bottom_y:
				bottom_y = y
		cv2.circle(image, (right_most_x, right_most_y), 1, (0, 255, 0), -1)
		cv2.circle(image, (left_most_x, left_most_y), 1, (0,255,0), -1)
		cv2.circle(image, ((right_most_x + left_most_x) /2, bottom_y), 1, (0,255,0), -1)
		cv2.circle(image, ((right_most_x + left_most_x)/2, top_y), 1, (0,255,0), -1)
		hght_mouth = bottom_y - top_y 
		wdth_mouth = right_most_x - left_most_x
		#print("Mouth height={}, width={}".format(hght_mouth, wdth_mouth))
	
		# Process the nose
		# Keep track of the top 4 points (smalled y)
		nose_y = []
		nose_x = []
		for (x,y) in facial_features["nose"]:
			nose_x.append(x)
			if len(nose_y) < 4:
				nose_y.append(y)
				nose_y.sort()
			else:
				if y < nose_y[3]:
					nose_y[3] = y
		cv2.circle(image, (sum(nose_x)/len(nose_x), nose_y[3]), 1, (0,255,0), -1)
		dist_mouth_nose = top_y - nose_y[3]
		#print("Distance nose to mouth={}".format(dist_mouth_nose))

		# Process the left eyebrow
		leb_x = []
		leb_y = []
		for (x,y) in facial_features["left_eyebrow"]:
			leb_x.append(x)
			leb_y.append(y)
		lAve_x = sum(leb_x)/len(leb_x)
		lAve_y = sum(leb_y)/len(leb_y)
		cv2.circle(image, (lAve_x, lAve_y), 1, (0,255,0), -1)

		# Process the right eyebrow
		reb_x = []
		reb_y = []
		for (x,y) in facial_features["right_eyebrow"]:
			reb_x.append(x)
			reb_y.append(y)
		rAve_x = sum(reb_x)/len(reb_x)
		rAve_y = sum(reb_y)/len(reb_y)
		cv2.circle(image, (rAve_x, rAve_y), 1, (0,255,0), -1)

	# Distance Calculations for between eybrows and eyes
	# Left Eye and Brow
	dist_l = math.sqrt(pow(lEye_x[0]-lAve_x, 2) + pow(lEye_y[0]-lAve_y, 2))
	#print dist_l
	# Right Eye and Brow
	dist_r = math.sqrt(pow(rEye_x[0]-rAve_x, 2) + pow(rEye_y[0]-rAve_y, 2))
	#print dist_r
	# Distance between Eyes
	dist_eyes = math.sqrt(pow(rEye_x[1]-lEye_x[1], 2) + pow(rEye_y[1]-lEye_y[1], 2))
	#print dist_eyes

	# All distances in a list
	distList = [dist_l, dist_r, dist_eyes, dist_mouth_nose, hght_mouth, wdth_mouth]
	
	# Average all the points/distances for each expression
	for key, d in EXPRESSION.items():
		if img_id in EXPRESSION_DICT[key].keys():
			if key == 'h':
				hList = [z + m for z, m in zip(hList, distList)]
				countH += 1
			elif key == 'sa':
				saList = [z + m for z, m in zip(saList, distList)]
				countSA += 1
			elif key == 'a':
				aList = [z + m for z, m in zip(aList, distList)]
				countA += 1
			elif key == 'n':
				nList = [z + m for z, m in zip(nList, distList)]
				countN += 1
			elif key == 'su':
				suList = [z + m for z, m in zip(suList, distList)]
				countSU += 1
			elif key == 'c':
				cList = [z + m for z, m in zip(cList, distList)]
				countC += 1
			elif key == 'o':
				oList = [z + m for z, m in zip(oList, distList)]
				countO += 1
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)

	#cv2.waitKey(0)

	if False:
	
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

# Does the averaging part
hList = [w / countH for w in hList]
saList = [w / countSA for w in saList]
aList = [w / countA for w in aList]
nList = [w / countN for w in nList]
suList = [w / countSU for w in suList]
cList = [w / countC for w in cList]
oList = [w / countO for w in oList]
AVE_EXPRESSION = {"h": hList, "sa": saList, "a": aList,\
		"n": nList, "su": suList, "c": cList, "o": oList} 
for key, value in AVE_EXPRESSION:
	print key, ":" 
	print "Distance of left eye & brow: ", value[0], "Distance of right eye & brow: ", value[1], \
		"Distance between eyes: ", value[2], "Distance from mouth to nose: ", value[3], \
		"Height of Mouth: ", value[4], "Width of Mouth: ", value[5] 




# Save classifications to JSON files
if False:
	for e, d in EXPRESSION_DICT.items():
		f = open(EXPRESSION[e] + ".pickle", "wb")
		pickle.dump(d, f)
		f.close()

if False:
	for img in glob.glob("projectImages/[K-M][0-9]*.bmp"): #THIS IS FOR TEST DATA
		#image = cv2.imread(args["image"])
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
		cv2.waitKey(1)
