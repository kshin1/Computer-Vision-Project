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

EXPRESSION = {"h":"happy", "sa": "sad", "a": "angry",\
				"n": "neutral", "su": "surprised", "c": "confused", "o": "other"}

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
	train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, random_state = 42, train_size = 0.7)
	print(test_X)
	# Use stored SVM
'''
	if os.path.isfile("svm.pickle"):
		f = open("svm.pickle", "rb")
		clf = pickle.load(f)
		f.close()
	# Create a SVM if it does not already exist
	else:
		clf = svm.SVC(kernel='poly',degree=3)

		# Fit the model using the training data
		clf.fit(train_X, train_Y)

		# Save our model to a pickle file 
		f = open("svm.pickle", "wb")
		pickle.dump(clf, f)
		f.close()

	# Use the model to predict the emotions of testing data
	predicted_Y = clf.predict(test_X)
	
	# Print accuracy of the model 
	print("Accuracy of the model: {}".format(str(sklmetrics.accuracy_score(test_Y, predicted_Y))))

	# Create a confusion matrix
	conf_mat = sklmetrics.confusion_matrix(test_Y, predicted_Y, labels = list(EXPRESSION.keys()))
	sns.heatmap(conf_mat, square=True, annot=True, cbar = False, xticklabels = list(EXPRESSION.keys()), yticklabels = list(EXPRESSION.keys()))
	plt.xlabel("Predicted Value")
	plt.ylabel("True Value")
	plt.title("Emotion Detection Accuracy")
	plt.show()
'''