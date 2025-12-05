#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    h = sigmoid(np.dot(data, W1) + b1)   # hidden layer
    y_hat = softmax(np.dot(h, W2) + b2)  # output layer 
    return y_hat[0, label]


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Forward propagation
    z1 = np.dot(data, W1) + b1                       # First layer linear transform
    h = sigmoid(z1)                                  # First layer activation
    z2 = np.dot(h, W2) + b2                          # Second layer linear transform
    y_hat = softmax(z2)                              # Second layer activation

    cost = -np.sum(labels * np.log(y_hat))           # Cross entropy loss

    # Backward propagation
    delta2 = y_hat - labels
    gradW2 = np.dot(h.T, delta2)
    gradb2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = np.dot(delta2, W2.T) * sigmoid_grad(h)
    gradW1 = np.dot(data.T, delta1)
    gradb1 = np.sum(delta1, axis=0, keepdims=True)

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Additional sanity checks.
    """
    print("Running your sanity checks...")

    # Small synthetic test network
    Dx, H, Dy = 4, 3, 4
    dimensions = [Dx, H, Dy]

    # One example
    data1 = np.random.randn(1, Dx)
    labels1 = np.zeros((1, Dy))
    labels1[0, np.random.randint(0, Dy)] = 1

    # Many examples
    M = 5
    dataM = np.random.randn(M, Dx)
    labelsM = np.zeros((M, Dy))
    for i in range(M):
        labelsM[i, np.random.randint(0, Dy)] = 1

    # Random parameters
    params = np.random.randn((Dx + 1) * H + (H + 1) * Dy)

    # --- Test 1: forward() shape test ---
    try:
        out1 = forward(data1, np.argmax(labels1), params, dimensions)
        print("Test 1 OK: forward() produces a scalar:", out1)
    except Exception as e:
        print("Test 1 FAILED:", e)

    # --- Test 2: softmax output sums to 1 ---
    h = sigmoid(np.dot(data1, np.reshape(params[:Dx*H], (Dx, H))) +
                np.reshape(params[Dx*H:Dx*H+H], (1, H)))
    logits = np.dot(h, np.reshape(params[Dx*H+H:Dx*H+H+H*Dy], (H, Dy))) + \
             np.reshape(params[-Dy:], (1, Dy))
    sm = softmax(logits)
    if np.allclose(np.sum(sm), 1.0):
        print("Test 2 OK: softmax sums to 1")
    else:
        print("Test 2 FAILED: softmax sum =", np.sum(sm))

    # --- Test 3: Run forward_backward_prop on multi-example input ---
    try:
        cost, grad = forward_backward_prop(dataM, labelsM, params, dimensions)
        print("Test 3 OK: forward_backward_prop runs. cost =", cost)
    except Exception as e:
        print("Test 3 FAILED:", e)

    # --- Test 4: Cost finite? ---
    if np.isfinite(cost):
        print("Test 4 OK: cost is finite")
    else:
        print("Test 4 FAILED: cost =", cost)

    # --- Test 5: Gradients finite? ---
    if np.all(np.isfinite(grad)):
        print("Test 5 OK: gradients are finite")
    else:
        print("Test 5 FAILED: grad contains NaN/Inf")

    # --- Test 6: Gradient checking on tiny network ---
    print("Running mini gradcheck (quick test)...")
    tiny_dims = [3, 2, 2]
    tiny_params = np.random.randn((3 + 1) * 2 + (2 + 1) * 2)
    tiny_data = np.random.randn(4, 3)
    tiny_labels = np.zeros((4, 2))
    for i in range(4):
        tiny_labels[i, np.random.randint(0, 2)] = 1

    try:
        gradcheck_naive(
            lambda p: forward_backward_prop(tiny_data, tiny_labels, p, tiny_dims),
            tiny_params
        )
        print("Test 6 OK: tiny gradcheck passed")
    except Exception as e:
        print("Test 6 FAILED:", e)

    # --- Test 7: Does forward() agree with forward_backward on 1 example? ---
    lbl = np.argmax(labels1)
    prob = forward(data1, lbl, params, dimensions)

    cost1, grad1 = forward_backward_prop(data1, labels1, params, dimensions)

    # y_hat[label] = exp(...) / sum(exp(...))
    # cost = -log(y_hat[label])
    if np.allclose(-np.log(prob), cost1):
        print("Test 7 OK: forward() agrees with forward_backward_prop()")
    else:
        print("Test 7 FAILED: forward vs backward mismatch")


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
