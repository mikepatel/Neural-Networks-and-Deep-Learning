# ECE 542
# Project 3: CNN
# October 2018

# dataset: MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labesl (digits 0-9)

# Notes:


################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

import os
from datetime import datetime

################################################################################
print("TF version: {}".format(tf.__version__))
print("GPU available: {}".format(tf.test.is_gpu_available()))


################################################################################
# loads MNIST data and
# returns (train_images, train_labels), (test_images, test_labels)
def load_mnist_data():
    # training images
    with open(os.path.join(os.getcwd(), "train-images-idx3-ubyte.gz"), "rb") as f:
        train_images = extract_images(f)

    # training labels
    with open(os.path.join(os.getcwd(), "train-labels-idx1-ubyte.gz"), "rb") as f:
        train_labels = extract_labels(f)

    # testing images
    with open(os.path.join(os.getcwd(), "t10k-images-idx3-ubyte.gz"), "rb") as f:
        test_images = extract_images(f)

    # testing labels
    with open(os.path.join(os.getcwd(), "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        test_labels = extract_labels(f)

    return (train_images, train_labels), (test_images, test_labels)


################################################################################
# loading dataset
(train_images, train_labels), (test_images, test_labels) = load_mnist_data()

# creating validation set from training set
# validation set size: 10k
valid_images = train_images[50000:]  # last 10k: 50k-60k
valid_labels = train_labels[50000:]  # last 10k: 50k-60k
train_images = train_images[:50000]  # first 50k
train_labels = train_labels[:50000]  # first 50k

# printing out shapes of sets
print("Training Images shape: {}".format(train_images.shape))
print("Training Labels shape: {}".format(train_labels.shape))
print("Validation Images shape: {}".format(valid_images.shape))
print("Validation Labels shape: {}".format(valid_labels.shape))
print("Testing Images shape: {}".format(test_images.shape))
print("Testing Labels shape: {}".format(test_labels.shape))

################################################################################
# HYPERPARAMETERS
NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_NEURONS_IN_DENSE_1 = 100

################################################################################
# input image dimensions
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)
'''
train_images = train_images.reshape(-1, img_rows, img_cols, num_channels)
valid_images = valid_images.reshape(-1, img_rows, img_cols, num_channels)
test_images = test_images.reshape(-1, img_rows, img_cols, num_channels)
'''

# output dimensions
num_classes = 10