#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', metavar='filename', type=str,
                        help='input file')

    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def test_network(file_name):
    print('testing network')
    print('loading saved file: ', file_name)
    model = network2.load(file_name)

    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()

    num_correct = 0
    for i in range(len(test_data[0])):
        prediction = np.argmax(model.feedforward(test_data[0][i]))
        if prediction == test_data[1][i]:
            num_correct += 1
    test_accuracy = num_correct / len(test_data[0])
    print("Test accuracy: " + str(test_accuracy))

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 20, 10])
    # train the network using SGD
    eval_cost,eval_accuracy,training_cost,training_accuracy = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    for k in range(len(training_accuracy)):
        training_accuracy[k] = training_accuracy[k] / len(train_data[0])
        eval_accuracy[k] = eval_accuracy[k] / len(valid_data[0])

    model.save("train.data")

    # Plot Learning curves
    print("PLOTING")
    #Training Loss
    plt.plot(list(range(0,len(model.training_loss_l))),model.training_loss_l)
    plt.title("Training Loss Curve")
    plt.savefig('training_loss')
    #plt.show()
    plt.clf()

    #Training accuracy
    plt.plot(list(range(0,len(training_accuracy))),training_accuracy)
    plt.title("Training Accuracy Curve")
    plt.savefig('training_accuracy')
    #plt.show()
    plt.clf()

    #Validation Loss
    plt.plot(list(range(0,len(model.validation_loss_l))),model.validation_loss_l)
    plt.title("Validation Loss Curve")
    plt.savefig('validation_loss')
    #plt.show()
    plt.clf()

    #Validation Accuracy
    plt.plot(list(range(0,len(eval_accuracy))),eval_accuracy)
    plt.title("Validation Accuracy Curve")
    plt.savefig('eval_accuracy')
    #plt.show()
    plt.clf()

    # Evaluate test data
    num_correct = 0
    for i in range(len(test_data[0])):
        prediction = np.argmax(model.feedforward(test_data[0][i]))
        if prediction == test_data[1][i]:
            num_correct += 1
    test_accuracy = num_correct / len(test_data[0])
    print("Test accuracy: " + str(test_accuracy))

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        test_network(FLAGS.test)
