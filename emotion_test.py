import pickle
import pandas as pd
import numpy as np
import sklearn as skl
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.metrics as sklmetrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import dlib
import imutils
from imutils import face_utils
import math

EXPRESSION = {"h":"happy", "sa": "sad", "a": "angry",\
				"n": "neutral", "su": "surprised", "c": "confused", "o": "other"}
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def classify_svm(img):
    features = extract_facial_distances(img)
    pred = clf.predict(features)

    # show classification result
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '{}'.format(pred), (10, 12), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return img

def extract_facial_distances(img):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	#image = cv2.imread(img)
	image = imutils.resize(img, width=500)
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
	facial_features = {}
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		facial_features[name] = shape[i:j]
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
	# Right Eye and Brow
	dist_r = math.sqrt(pow(rEye_x[0]-rAve_x, 2) + pow(rEye_y[0]-rAve_y, 2))
	# Distance between Eyes
	dist_eyes = math.sqrt(pow(rEye_x[1]-lEye_x[1], 2) + pow(rEye_y[1]-lEye_y[1], 2))

	# All distances in a list
	all_dist = [{"dist_l": dist_l, "dist_r": dist_r, "dist_eyes": dist_eyes, "dist_mouth_nose": dist_mouth_nose, "hght_mouth": hght_mouth, "wdth_mouth": wdth_mouth}]
	return pd.DataFrame(all_dist)

def camera_loop():
    print("Press <SPACE> to capture/classify an image, or <Esc> to exit.")
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()

        action = cv2.waitKey(1)

        cv2.imshow('camera', frame)

        if action == ord('q') or action == 27:
            break

        if action == ord(' '):
            # svm object detection
            frame = classify_svm(frame)
            cv2.imshow('SVM output:', frame)

    cap.release()

if __name__ == "__main__":
	data = pd.read_csv("emotions.csv")
	data_X = data.drop("emotion", axis = 1)
	data_Y = data["emotion"]

	# Split to training and testing data
	#train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, random_state = 42, train_size = 0.7)

	# Use stored SVM
	if os.path.isfile("svm.pickle"):
		f = open("svm.pickle", "rb")
		clf = pickle.load(f)
		f.close()
	# Create a SVM if it does not already exist
	else:
		clf = svm.SVC(kernel='poly',degree=3)

		# Fit the model using the training data
		clf.fit(data_X, data_Y)

		# Save our model to a pickle file 
		f = open("svm_2.pickle", "wb")
		pickle.dump(clf, f)
		f.close()
	camera_loop()
	cv2.destroyAllWindows()
