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
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

import os
from datetime import datetime
import matplotlib.pyplot as plt

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

'''
print("\n@@@@@@@@@@@@@@@@@@")
print(train_labels[:10])
print(train_labels[5555:5565])
print(test_labels[:10])
print(test_labels[8888:8898])
'''

# creating validation set from training set
# validation set size: 10k
valid_images = train_images[50000:]  # last 10k: 50k-60k
valid_labels = train_labels[50000:]  # last 10k: 50k-60k
train_images = train_images[:50000]  # first 50k
train_labels = train_labels[:50000]  # first 50k

# printing out shapes of sets
print("\n##########")
print("Training Images shape: {}".format(train_images.shape))
print("Training Labels shape: {}".format(train_labels.shape))
print("Validation Images shape: {}".format(valid_images.shape))
print("Validation Labels shape: {}".format(valid_labels.shape))
print("Testing Images shape: {}".format(test_images.shape))
print("Testing Labels shape: {}".format(test_labels.shape))
print("\n##########")

################################################################################
# HYPERPARAMETERS
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_NEURONS_IN_DENSE_1 = 128
DROP_PROB = 0.5

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

################################################################################
# build model
model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=[5, 5],
    input_shape=input_shape,
    padding="same",
    activation=relu
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
    activation=relu
))

model.add(BatchNormalization())

model.add(MaxPool2D(
    pool_size=[2, 2],
    strides=2
))

model.add(Flatten())

model.add(Dense(
    units=NUM_NEURONS_IN_DENSE_1,
    activation=relu
))

model.add(Dropout(DROP_PROB))

model.add(Dense(
    units=num_classes,
    activation=softmax
))

# configure model for training
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

model.summary()

################################################################################
# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
dir = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
history_file = dir + "\cnn.h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=dir)

# train model
history = model.fit(
    x=train_images,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(valid_images, valid_labels),
    callbacks=[save_callback, tb_callback],
    verbose=0
)

#
history_dict = history.history
train_accuracy = history_dict["acc"]
train_loss = history_dict["loss"]
valid_accuracy = history_dict["val_acc"]
valid_loss = history_dict["val_loss"]

# predictions / evaluation on test set
test_loss, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels,
    verbose=0
)

################################################################################
# Visualization and Output
num_epochs_plot = range(1, len(train_accuracy) + 1)

# Loss curves
plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy curves
plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Test loss and accuracy
print("\n##########")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))
print("##########")
