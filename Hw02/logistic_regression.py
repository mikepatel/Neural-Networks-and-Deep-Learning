#! /usr/bin/env python3

# Michael Patel
# mrpatel5
# ECE 542
# Fall 2018
# Hw02

# LOGISTIC REGRESSION
# Apply logistic regression and random forest on MNIST dataset
# use raw pixel values as features
# predict digit labels based on images

# dataset: MNIST
# training size: 60k
# testing size: 10k
# 10 class labels: 0-9

# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/Logistic_Regression/LogisticRegression_MNIST_Codementor.ipynb

###############################################################################
# IMPORTs
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

###############################################################################
# load dataset
mnist = fetch_mldata("MNIST original")
train_images, test_images, train_labels, test_labels = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0
)

# explore data
#print(train_images.shape)  # (60k, 784)
#print(train_labels[0:5])

# instantiate model
logReg = LogisticRegression(solver="lbfgs")

# training
logReg.fit(train_images, train_labels)

# predictions
predictions = logReg.predict(test_images)
#print(predictions[0:10])

# measuring model performance
accuracy = logReg.score(test_images, test_labels)
#print(accuracy)

# save predictions to csv
prob = logReg.predict_proba(test_images)  # prob for each class label
prob = np.round(prob)  # push prob to 0 or 1
prob = prob.astype(int)
#print(type(prob))
#print(prob[0:10])

df = pd.DataFrame(prob)
df.to_csv("lr.csv", header=None, index=None)