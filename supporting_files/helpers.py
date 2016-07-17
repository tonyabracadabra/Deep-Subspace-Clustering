from __future__ import print_function
from sda import StackedDenoisingAutoencoder

import tensorflow as tf
import numpy as np

def activate(layer, name):
    if name == 'sigmoid':
        return tf.nn.sigmoid(layer)
    elif name == 'softmax':
        return tf.nn.softmax(layer)
    elif name == 'tanh':
        return tf.nn.tanh(layer)
    elif name == 'relu':
        return tf.nn.relu(layer)
    elif name == 'linear':
        return layer

def optimize(cost, learning_rate, optimizer):
    optimizer = {'FTRL':tf.train.FtrlOptimizer, 'Adam':tf.train.AdamOptimizer, \
                 'SGD':tf.train.GradientDescentOptimizer}[optimizer]

    return optimizer(learning_rate=learning_rate).minimize(cost)

def one_hot(y):
    n_classes = len(np.unique(y))
    one_hot_Y = np.zeros((len(y), n_classes))
    for i,j in enumerate(y):
        one_hot_Y[i][j] = 1
        
    return one_hot_Y

def init_layer_weight(dims, X, name):
    weights, biases = [], []
    if name == 'sda':
        sda = StackedDenoisingAutoencoder(dims=dims)
        sda._fit(X)
        weights, biases = sda.weights, sda.biases
    elif name == 'uniform':
        n_in = X.shape[1]
        for d in dims:
            r = 4*np.sqrt(6.0/(n_in+d))
            weights.append(tf.random_uniform([n_in, d], minval=-r, maxval=r))
            biases.append(tf.zeros([d,]))
            n_in = d
            
    return weights, biases
    
def get_batch(X, Y, size):
    assert len(X) == len(Y)
    a = np.random.choice(len(X), size, replace=False)
    return X[a], Y[a]

def get_batch_XC(X, C, size):
    assert len(X) == len(C)
    a = np.random.choice(len(X), size, replace=False)
    return X[a], C[a, a]

class GenBatch():
    """
        class for generating batches
    """
    def __init__(self, X, y=None, C=None, batch_size=500):
        self.X = X
        self.Y = y
        self.C = C
        self.batch_size = batch_size
        self.n_batch = (len(X) / batch_size)
        self.index = 0

    def get_batch(self):
        start, end = self.index, (self.index+1)*self.batch_size
        batch_range = xrange(start, end)
        if self.index == self.n_batch:
            end = len(self.X)
            batch_range = xrange(self.index, len(self.X))
        self.index += 1

        return_list = [self.X[batch_range]]
        if self.Y is not None:
            return_list.append(self.Y[batch_range])
        if self.C is not None:
            return_list.append(self.C[start:end, start:end])

        return return_list

    def resetIndex(self):
        self.index = 0