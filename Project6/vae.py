# ECE 542
# Project 6
# Fall 2018

# Variational Autoencoder

# dataset: MNIST

# Notes:
#   - see how MNIST data cluster in lower-dimensional space according to their digit class
#   - keras -> wrap code not part of built-in layer into Lambda layer
#   - 2 loss functions: reconstruction loss, regularization loss

################################################################################
# IMPORTs
import os
import numpy

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Lambda, Conv2D, Flatten \
    Dropout, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


################################################################################
# load data
def load_data():
    cwd = os.getcwd()
    # training images
    with open(os.path.join(cwd, "train-images-idx3-ubyte.gz"), "rb") as f:
        train_images = extract_images(f)

    # training labels
    with open(os.path.join(cwd, "train-labels-idx1-ubyte.gz"), "rb") as f:
        train_labels = extract_labels(f)

    # testing images
    with open(os.path.join(cwd, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        test_images = extract_images(f)

    # testing labels
    with open(os.path.join(cwd, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        test_labels = extract_labels(f)

    return (train_images, train_labels), (test_images, test_labels)


################################################################################
# HYPERPARAMETERS and DESIGN CHOICES
BATCH_SIZE = 64
NUM_EPOCHS = 10
LATENT_DIM = 2  # 2-dimensional to construct plane

# MNIST dimensions
num_rows = 28
num_cols = 28
num_channels = 1

################################################################################
# build encoder
def build_encoder():
    m = Sequential()

    m.add(Conv2D(
        filters=32,
        kernel_size=[4, 4],
        input_shape=(num_rows, num_cols, num_channels),
        padding="same",
        activation=relu
    ))

    m.add(Conv2D(
        filters=64,
        kernel_size=[4, 4],
        strides=2,
        padding="same",
        activation=relu
    ))

    m.add(Conv2D(
        filters=64,
        kernel_size=[4, 4],
        padding="same",
        activation=relu
    ))

    m.add(Conv2D(
        filters=64,
        kernel_size=[4, 4],
        padding="same",
        activation=relu
    ))

    return m


################################################################################
# build decoder
def build_decoder():
    m = Sequential()

    return m


################################################################################
# get data
# instantiate model

# TRAINING

################################################################################
# VISUALIZATION
