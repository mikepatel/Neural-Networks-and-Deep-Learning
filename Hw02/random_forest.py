#! /usr/bin/env python3

# Michael Patel
# mrpatel5
# ECE 542
# Fall 2018
# Hw02

# RANDOM FOREST
# Apply logistic regression and random forest on MNIST dataset
# use raw pixel values as features
# predict digit labels based on images

# dataset: MNIST
# training size: 60k
# testing size: 10k
# 10 class labels: 0-9

# http://brianfarris.me/static/digit_recognizer.html

###############################################################################
# IMPORTs
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

###############################################################################
# load dataset
mnist = fetch_mldata("MNIST original")
train_images, test_images, train_labels, test_labels = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0
)

# explore data
#print(train_labels[0:5])

# instantiate model
rf = RandomForestClassifier()

# training
rf.fit(train_images, train_labels)

# predictions
predictions = rf.predict_proba(test_images)  # prob for each class label
predictions = np.round(predictions)  # push to 0 or 1
predictions = predictions.astype(int)
#print(predictions[0:5])

# measuring model performance
accuracy = rf.score(test_images, test_labels)

# save predictions to csv
df = pd.DataFrame(predictions)
df.to_csv("rf.csv", header=None, index=None)
