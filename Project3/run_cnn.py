# ECE 542
# Project 3: CNN
# October 2018
# Description: This script trains and tests the final CNN after hyper-parameter optimization.

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
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cnn

################################################################################
print("TF version: {}".format(tf.__version__))
print("GPU available: {}".format(tf.test.is_gpu_available()))


################################################################################

# loading dataset
(train_images, train_labels), (test_images, test_labels) = cnn.load_mnist_data()

'''
print("\n@@@@@@@@@@@@@@@@@@")
print(train_labels[:10])
print(train_labels[5555:5565])
print(test_labels[:10])
print(test_labels[8888:8898])
'''

# creating validation set from training set
# validation set size:
valid_set_size = 8000
split = len(train_images) - valid_set_size
valid_images = train_images[split:]  # last
valid_labels = train_labels[split:]  # last
train_images = train_images[:split]  # first
train_labels = train_labels[:split]  # first

# printing out shapes of sets
'''
print("\n##########")
print("Training Images shape: {}".format(train_images.shape))
print("Training Labels shape: {}".format(train_labels.shape))
print("Validation Images shape: {}".format(valid_images.shape))
print("Validation Labels shape: {}".format(valid_labels.shape))
print("Testing Images shape: {}".format(test_images.shape))
print("Testing Labels shape: {}".format(test_labels.shape))
print("\n##########")
'''

################################################################################
# HYPERPARAMETERS AND DESIGN CHOICES
NUM_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_NEURONS_IN_DENSE_1 = 256
DROP_PROB = 0.6
ACTIV_FN = "relu"
activation_fn = cnn.get_activ_fn(ACTIV_FN)

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
# callbacks for Save weights, Tensorboard
# creating a new directory for each run using timestamp
folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"), str(ACTIV_FN))
history_file = folder + "\cnn_" + str(ACTIV_FN) + ".h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=folder)

# Build, train, and test model
model = cnn.build_model(input_shape, activation_fn, LEARNING_RATE, DROP_PROB, NUM_NEURONS_IN_DENSE_1, num_classes)
train_accuracy, train_loss, valid_accuracy, valid_loss = cnn.train_model(model, train_images, train_labels, BATCH_SIZE,
                                                                     NUM_EPOCHS, valid_images, valid_labels,
                                                                     save_callback, tb_callback)
test_accuracy, test_loss, predictions = cnn.test_model(model, test_images, test_labels)

# save test set results to csv
predictions = np.round(predictions)
predictions = predictions.astype(int)
df = pd.DataFrame(predictions)
df.to_csv("mnist.csv", header=None, index=None)

################################################################################
# Visualization and Output
num_epochs_plot = range(1, len(train_accuracy) + 1)

# Loss curves
plt.figure(1)
plt.plot(num_epochs_plot, train_loss, "b", label="Training Loss")
plt.plot(num_epochs_plot, valid_loss, "r", label="Validation Loss")
plt.title("Loss Curves_" + ACTIV_FN)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Figures/' + ACTIV_FN + '_loss.png')
plt.show()

# Accuracy curves
plt.figure(2)
plt.plot(num_epochs_plot, train_accuracy, "b", label="Training Accuracy")
plt.plot(num_epochs_plot, valid_accuracy, "r", label="Validation Accuracy")
plt.title("Accuracy Curves_" + ACTIV_FN)
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Figures/' + ACTIV_FN + '_acc.png')
plt.show()

# Test loss and accuracy
print("\n##########")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))
print("##########")