#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    temp = (np.transpose(((np.transpose(weights[0]))*x))) + biases[0]
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    # Feed forward code. @TODO still need to figure out what to pass the cost delta function - Zach Johnston    
    h = []
    h.append(x)
    #print(h[0].shape)
    a = []
    for k in range(1,num_layers):
        a.append(np.dot(weights[k-1],h[k-1]) + biases[k-1])
        h.append(sigmoid(a[k-1]))

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    #delta = (cost).delta(activations[-1], y)
    delta = (cost).delta(h[-1], y)
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    #for layer in range((num_layers-1),0,-1): # Backpropagate the error
    #    error_prev = np.multiply(np.dot(np.transpose(weights[layer-1]),error_prev),sigmoid_prime(h[layer-1]))
    #    error.append(error_prev)
    #    nabla_b.append(error_prev)
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta,h[-2].transpose())
    for layer in range(2,num_layers):
        delta = np.dot(weights[-layer+1].transpose(),delta) * sigmoid_prime(a[-layer])
        nabla_b[-layer] = delta
        nabla_w[-layer] = np.dot(delta,h[-layer-1].transpose())


    return (nabla_b, nabla_w)

