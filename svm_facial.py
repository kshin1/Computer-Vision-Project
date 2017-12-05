import pickle
import pandas as pd
import numpy as np
import sklearn as skl
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.metrics as skimetrics

data = pd.read_csv("emotions.csv")
data_X = data.drop("emotion", axis = 1)
data_Y = data["emotion"]

# Split to training and testing data
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, random_state = 42, train_size = 0.7)

# Create a SVM 
clf = svm.SVC()

# Fit the model using the training data
clf.fit(train_X, train_Y)

# Use the model to predict the emotions of testing data
predicted_Y = clf.predict(test_X)

# Print accuracy of the model 
print skimetrics.accuracy_score(test_Y, predicted_Y)
