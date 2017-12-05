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
	if os.path.isfile("svm_2.pickle"):
		f = open("svm_2.pickle", "rb")
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
