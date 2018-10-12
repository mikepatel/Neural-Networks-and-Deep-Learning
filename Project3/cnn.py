# ECE 542
# Project 3: CNN
# October 2018
# Description: This script contains all of the functions used for loading data, building, training, and testing CNN

# dataset: MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labesl (digits 0-9)

# Notes:
#   - validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data

################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import os
import numpy as np



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
# assigns the specified activation function
def get_activ_fn(s):
    if s == "relu":
        return relu
    elif s == "softmax":
        return softmax
    elif s == "tanh":
        return tanh
    elif s == "sigmoid":
        return sigmoid


################################################################################
# build model
def build_model(input_shape, activation_fn, LEARNING_RATE, DROP_PROB, NUM_NEURONS_IN_DENSE_1, num_classes):
    model = Sequential()

    model.add(Conv2D(
        filters=32,
        kernel_size=[5, 5],
        input_shape=input_shape,
        padding="same",
        activation=activation_fn
    ))

    model.add(BatchNormalization())

    model.add(MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    model.add(Conv2D(
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=activation_fn
    ))

    model.add(BatchNormalization())

    model.add(MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    model.add(Flatten())

    model.add(Dense(
        units=NUM_NEURONS_IN_DENSE_1,
        activation=activation_fn
    ))

    model.add(Dropout(DROP_PROB))

    model.add(Dense(
        units=num_classes,
        activation=softmax
    ))

    # configure model for training
    # i.e. define loss function, optimizer, training metrics
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(lr=LEARNING_RATE),
        metrics=["accuracy"]
    )

    model.summary()
    return model


############################################################################
# train model
def train_model(model, train_images, train_labels, BATCH_SIZE, NUM_EPOCHS, valid_images, valid_labels, save_callback, tb_callback):
    history = model.fit(
        x=train_images,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(valid_images, valid_labels),
        shuffle=True,
        callbacks=[save_callback, tb_callback],
        verbose=0
    )
    history_dict = history.history
    train_accuracy = history_dict["acc"]
    train_loss = history_dict["loss"]
    valid_accuracy = history_dict["val_acc"]
    valid_loss = history_dict["val_loss"]
    return train_accuracy, train_loss, valid_accuracy, valid_loss


#################################################################################
# evaluation on test set
def test_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(
        x=test_images,
        y=test_labels,
        verbose=0
    )
    # predictions with test set
    predictions = model.predict_proba(
        x=test_images,
        batch_size=None,
        verbose=0
    )
    return test_accuracy, test_loss, predictions



