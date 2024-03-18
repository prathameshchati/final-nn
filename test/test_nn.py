# TODO: import dependencies and write unit tests below
import numpy as np
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import re
import random
from nn import io, preprocess
from nn.nn import NeuralNetwork
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# for the tests, we can primarly test if the outputted dimensions are correct for each run given a single epoch

# load in the digits dataset
digits=load_digits()

# get data (normalize with max value such that values range from 0 to 1) and the values
digits_data=digits['data']
digits_data=digits_data/digits_data.max()
digits_target=digits['target']

# generate train and test sets
X_train, X_val, y_train, y_val=train_test_split(np.array(digits_data), np.array(digits_target), test_size=0.3, random_state=543)


def test_single_forward():
    # initialize autoencoder architecture
    nn_arch=[{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'}, {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}] # best: sigmoid, relu

    # initialize neural network and set hyperparameters
    lr=0.1
    seed=343
    batch_size=200
    epochs=1 # single epoch
    loss_function='mse'
    nn=NeuralNetwork(nn_arch, lr=lr, seed=seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function) # loss function can be mse or bce

    # split into batches for testing
    X_train_batches=[X_train[i:i + nn._batch_size] for i in range(0, len(X_train), nn._batch_size)]  
    y_train_batches=[y_train[i:i + nn._batch_size] for i in range(0, len(y_train), nn._batch_size)]  

    # for the first layer (1), A_prev is set to X_train, W1 should have dimensions output layer by input layer (16 x 64), b1 should have dimensions output layer (16,1), A_prev should have dimensions batch_size by features (200 x 64)
    layer_idx=1
    A_prev=X_train_batches[0]
    assert A_prev.shape==(200,64)
    assert nn._param_dict['W' + str(layer_idx)].shape==(16,64) 
    assert nn._param_dict['b' + str(layer_idx)].shape==(16,1)

    # once the dimensions above match, we can run the method using these inputs for the first layer
    A_curr, Z_curr=nn._single_forward(nn._param_dict['W' + str(layer_idx)], nn._param_dict['b' + str(layer_idx)], A_prev, nn_arch[0]['activation'])

    # both A_curr and Z_curr should have the same dimensions, which should be batch_size by output layer (200 x 16)
    assert A_curr.shape==Z_curr.shape
    assert A_curr.shape==(200,16)
    assert A_curr.shape==(200,16)

    # the second layer (2), A_prev is set to A_curr, and W2 should have dimensions output layer by input layer (64 x 16), and b1 should have dimensions (64,1)
    layer_idx=2
    A_prev=A_curr
    assert nn._param_dict['W' + str(layer_idx)].shape==(64,16) 
    assert nn._param_dict['b' + str(layer_idx)].shape==(64,1)

    # run the method again 
    A_curr, Z_curr=nn._single_forward(nn._param_dict['W' + str(layer_idx)], nn._param_dict['b' + str(layer_idx)], A_prev, nn_arch[1]['activation'])

    # similarly, A_curr and Z_curr should have the same dimensions, batch_size by output layer  (200,64)
    assert A_curr.shape==Z_curr.shape
    assert A_curr.shape==(200,64)
    assert A_curr.shape==(200,64)

    # pass

def test_forward():

    # similar to above, we can manually enter the initial inputs to forward and ensure the output A_curr and stored matrices in cache are of the correct dimensions

    # initialize autoencoder architecture
    nn_arch=[{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'}, {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}] # best: sigmoid, relu

    # initialize neural network and set hyperparameters
    lr=0.1
    seed=343
    batch_size=200
    epochs=1 # single epoch
    loss_function='mse'
    nn=NeuralNetwork(nn_arch, lr=lr, seed=seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function) # loss function can be mse or bce

    # split into batches for testing
    X_train_batches=[X_train[i:i + nn._batch_size] for i in range(0, len(X_train), nn._batch_size)]  
    y_train_batches=[y_train[i:i + nn._batch_size] for i in range(0, len(y_train), nn._batch_size)]  

    # run forward with first batch
    A_curr, cache=nn.forward(X_train_batches[0])

    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass