import math
import pickle
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import os
# distList = [dist_l, dist_r, dist_eyes, dist_mouth_nose, hght_mouth, wdth_mouth]
def angryCalc(diffArry):
	score = 0
	# Distance of left eye & brow
	if 4 <= diffArry[0] <= 7:
		score += 1#0.175
	# Distance of right eye & brow
	if 4 <= diffArry[1] <= 7:
		score += 1#0.175
	# Distance from mouth to nose
	if -5 <= diffArry[3] <= 5:
		score += 1#0.15
	# Height of mouth
	if 3 <= diffArry[4] <= 15:
		score += 1#0.25
	# Width of mouth
	if 1 <= diffArry[5] <= 12:
		score += 1#0.25
	return score

def confusedCalc(diffArry): 
	score = 0
	# Distance of left eye & brow
	if diffArry[0] <= -3:
		score += 1#0.2
	# Distance of right eye & brow
	if diffArry[1] <= -3:
		score += 1#0.2
	# Distance from mouth to nose
	if 4 <= diffArry[3] <= 14:
		score += 1#0.2
	# Height of mouth
	if -1 <= diffArry[4] <= 10:
		score += 1#0.2
	# Width of mouth
	if 15 <= diffArry[5] <= 25:
		score += 1#0.2
	return score

def happyCalc(diffArry):
	score = 0
	# Distance of left eye & brow
	if -4 <= diffArry[0] <= 2:
		score += 1#0.075
	# Distance of right eye & brow
	if -4 <= diffArry[1] <= 2:
		score += 1#0.075
	# Distance from mouth to nose
	if diffArry[3] >= 15:
		score += 1#0.35
	# Height of mouth
	if -25 <= diffArry[4] <= -15:
		score += 1#0.2
	# Width of mouth
	if diffArry[5] <= -3:
		score += 1#0.3
	return score
def surprisedCalc(diffArry):
    score = 0
    # Distance of left eye & brow
    if 5 <= diffArry[0] <= 15:
        score += 1#.25
    # Distance of right eye & brow
    if 5 <= diffArry[1] <= 15:
        score += 1#.25
    # Distance from mouth to nose
    if -5<= diffArry[3] <= 5:
        score += 1#.125
    # Height of mouth
    if -15 <= diffArry[4] < -25:
        score += 1#.125
    # Width of mouth
	if 15 <= diffArry[5] <= 25:
		score += 1#.25
    return score
 
def neutralCalc(diffArry):
    score = 0
    # Distance of left eye & brow
    if diffArry[0] < 10 and diffArry[0] > -10:
        score += 1#.2
    # Distance of right eye & brow
    if diffArry[1] < 10 and diffArry[1] > -10:
        score += 1#.2
    # Distance from mouth to nose
    if diffArry[3] < 10 and diffArry[3] > -10:
        score += 1#.2
    # Height of mouth
    if diffArry[4] < 10 and diffArry[4] > -10:
        score += 1#.2
    # Width of mouth
    if diffArry[5] < 10 and diffArry[5] > -10:
        score += 1#.2
 
    return score
 
def sadCalc(diffArry):
    score = 0
    # Distance of left eye & brow
    if diffArry[0] > -5 and diffArry[0] < 2:
        score += 1#.2
    # Distance of right eye & brow
    if diffArry[1] > -5 and diffArry[1] < 2:
        score += 1#.2
    # Distance from mouth to nose
    if diffArry[3] > 3 and diffArry[3] < 13:
        score += 1#.2
    # Height of mouth
    if diffArry[4] > -11 and diffArry[4] < 0:
        score += 1#.2
    # Width of mouth
    if diffArry[5] < 5 and diffArry[5] > -5:
        score += 1#.2
 
    return score

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

for key, d in EXPRESSION_DICT.items():
	if os.path.isfile(EXPRESSION[key] + ".pickle"):
		f = open(EXPRESSION[key] + ".pickle", "rb")
		EXPRESSION_DICT[key] = pickle.load(f)
		f.close()

f = open("average_dict.pickle", "rb")
avg_dist = pickle.load(f)
f.close()

# Calculate the difference
avg_diff = {}
for emotion, diffs in avg_dist.items():
	avg_diff[emotion] = [ avg_dist["n"][i] - diffs[i] for i in range(len(diffs)) ]

correct = 0
total = 0

actual_count = {}
correct_count = {}
for e in EXPRESSION.keys():
	actual_count[e] = 0
	correct_count[e] = 0

# Testing 
for img in glob.glob("projectImages/[A-J][0-9]*.bmp"):
	img_id = img.split("/")[1]

	# Get actual emotion and the facial_features extracted
	for key, d in EXPRESSION_DICT.items():
		if img_id in d.keys():
			actual = key
			facial_features = d[img_id]
			break

	# Calculate points of interest
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

	# Process the right eyebrow
	reb_x = []
	reb_y = []
	for (x,y) in facial_features["right_eyebrow"]:
		reb_x.append(x)
		reb_y.append(y)
	rAve_x = sum(reb_x)/len(reb_x)
	rAve_y = sum(reb_y)/len(reb_y)

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

	# Calculate difference to neutral expression
	calcDiff = [ distList[i] - avg_dist["n"][i] for i in range(len(distList)) ]

	# Determine which emotion
	pointDict = {}
	pointDict["a"] = angryCalc(calcDiff)
	pointDict["c"] = confusedCalc(calcDiff)
	pointDict["h"] = happyCalc(calcDiff)
	pointDict["su"] = surprisedCalc(calcDiff)
	pointDict["n"] = neutralCalc(calcDiff)
	pointDict["sa"] = sadCalc(calcDiff)
	
	max_pt = -10
	predicted = None
	for emot, point in pointDict.items():
		if point > max_pt:
			predicted = emot 
			max_pt = point

	#print(pointDict)

	print("{}\tPredicted: {}, Actual: {}".format(img_id, EXPRESSION[predicted], EXPRESSION[actual]))
	# Select the best emotion
	if actual == predicted:
		correct_count[actual] = correct_count.get(actual, 0) + 1
		correct += 1
	total += 1
	actual_count[actual] = actual_count.get(actual, 0) + 1

print("Tested {} images, {} correct".format(total, correct))
print("Overall percent accuracy: {}%".format((float(correct) / total)* 100))
for key, value in EXPRESSION.items():
	if actual_count[key] == 0:
		perc = 0
	else:
		perc = float(correct_count[key])/actual_count[key]
	print("\t{}: {}/{} correct \tPercentage: {}%".format(value, correct_count[key], actual_count[key], perc * 100))
