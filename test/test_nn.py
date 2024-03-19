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
import tensorflow as tf

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
    assert Z_curr.shape==(200,16)
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
    assert Z_curr.shape==(200,64)
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

    # A_curr should hagve dimensions of batch_size by last layer (200,64)
    assert A_curr.shape==(200,64)

    # check sizes of cache, which should match above
    assert cache['A0'].shape==(200,64)
    assert cache['A1'].shape==(200,16)
    assert cache['Z1'].shape==(200,16)
    assert cache['Z2'].shape==(200,64)

    # pass

def test_single_backprop():

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

    # first get the cache using forward and y_hat using predict
    A_curr, cache=nn.forward(X_train_batches[0])
    y_hat=nn.predict(X_train_batches[0])

    # get inputs to backprop to test _single_backprop - start at last layer (2)
    layer_idx=2
    dA_curr=nn._mean_squared_error_backprop(y_train_batches[0].reshape(len(y_train_batches[0]), 1), y_hat)

    # run method on last layer, check dimensions of outputs
    dA_prev, dW_curr, db_curr=nn._single_backprop(nn._param_dict['W'+str(layer_idx)], nn._param_dict['b'+str(layer_idx)], cache['Z'+str(layer_idx)], cache['A'+str(layer_idx-1)], dA_curr, nn_arch[layer_idx-1]['activation'])

    # check that the outputs grad_dict have correct dimensions
    assert dA_prev.shape==(200,16) # dims are batch_size by prev layer
    assert dW_curr.shape==(64,16) # output dim by input dim (if looking forward)
    assert db_curr.shape==(64,1) # output dim by 1 (if looking forward)

    # for the next layer, set dA_curr to dA_prev
    dA_curr=dA_prev

    # get inputs to backprop to test _single_backprop - start at middle layer (1)
    layer_idx=1

    # run method on middle layer, check dimensions of outputs
    dA_prev, dW_curr, db_curr=nn._single_backprop(nn._param_dict['W'+str(layer_idx)], nn._param_dict['b'+str(layer_idx)], cache['Z'+str(layer_idx)], cache['A'+str(layer_idx-1)], dA_curr, nn_arch[layer_idx-1]['activation'])

    # check that the outputs grad_dict have correct dimensions
    assert dA_prev.shape==(200,64) # dims are batch_size by prev layer
    assert dW_curr.shape==(16,64) # output dim by input dim (if looking forward)
    assert db_curr.shape==(16,1) # output dim by 1 (if looking forward)

    # pass

def test_predict():
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

    # call predict
    y_hat=nn.predict(X_train_batches[0])

    # check that the dimensions are the batch_size by 1
    assert y_hat.shape==(200,1)

    # pass

def test_binary_cross_entropy():
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

    # call predict
    y_hat=nn.predict(X_train_batches[0])

    # get our bce and tensorflows bce
    nn_bce=nn._binary_cross_entropy(y_train_batches[0], y_hat)
    bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    tf_bce=bce(y_train_batches[0], y_hat).numpy()

    # make sure values are roughly equal
    assert np.isclose(nn_bce, tf_bce, atol=0.01)

    # pass

def test_binary_cross_entropy_backprop():
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

    # call predict
    y_hat=nn.predict(X_train_batches[0])

    # call the bce backprop
    dA_curr=nn._binary_cross_entropy_backprop(y_train_batches[0].reshape(len(y_train_batches[0]), 1), y_hat)

    # check that the outputs are of the right size
    assert dA_curr.shape==(200,1)
    
    # pass

def test_mean_squared_error():
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

    # call predict
    y_hat=nn.predict(X_train_batches[0])

    # get our mse and tensorflows mse
    nn_mse=nn._mean_squared_error(y_train_batches[0], y_hat)
    mse=tf.keras.losses.MeanSquaredError()
    tf_mse=mse(y_train_batches[0], y_hat).numpy()

    # make sure values are roughly equal
    assert np.isclose(nn_mse, tf_mse, atol=0.01)

    # pass

def test_mean_squared_error_backprop():
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

    # call predict
    y_hat=nn.predict(X_train_batches[0])

    # call the mse backprop
    dA_curr=nn._mean_squared_error_backprop(y_train_batches[0].reshape(len(y_train_batches[0]), 1), y_hat)

    # check that the outputs are of the right size
    assert dA_curr.shape==(200,1)

    # pass

def test_sample_seqs():
    # get positive and negative sequences, split negative sequences into k-mers
    pos_seqs=io.read_text_file("./data/rap1-lieb-positives.txt")
    neg_seqs=io.read_fasta_file("./data/yeast-upstream-1k-negative.fa")

    # split each negative sequence into kmers 
    seq_length=len(pos_seqs[0])
    neg_seqs_kmers=[]
    for neg_seq in neg_seqs:
        for idx, _ in enumerate(neg_seq):
            neg_seq_kmer=neg_seq[idx:idx+seq_length]
            if (len(neg_seq_kmer)==seq_length):
                neg_seqs_kmers.append(neg_seq_kmer)
            else:
                break

    # run the sample_seqs method and see if the number of positive and negative labels match
    # aggregate all seqs and labels and sample
    all_seqs=pos_seqs+neg_seqs_kmers
    all_labels=[True]*len(pos_seqs)+[False]*len(neg_seqs_kmers)
    sampled_seqs, sampled_labels=preprocess.sample_seqs(all_seqs, all_labels)

    # check that the number of positive labels is all of the positive labels
    num_pos=0
    num_neg=0
    for label in sampled_labels:
        if (label==True):
            num_pos+=1
        else:
            num_neg+=1
    
    assert num_pos==num_neg

    # pass

def test_one_hot_encode_seqs():

    """
    Encodings:
        A -> [1, 0, 0, 0]
        T -> [0, 1, 0, 0]
        C -> [0, 0, 1, 0]
        G -> [0, 0, 0, 1]
        
    Example: AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].

    """

    # input test sequences and check manually
    seqs=['ATCG', 'AAAA', 'TTTT', 'CCCC', 'GGGG']

    # account for edge cases and empty strings
    seqs_encoded=preprocess.one_hot_encode_seqs(seqs)

    # check outputs and lengths of outputs
    atcg=[1,0,0,0]+[0,1,0,0]+[0,0,1,0]+[0,0,0,1]
    atcg=list(map(float, atcg))
    assert seqs_encoded[0]==atcg
    assert seqs_encoded[1]==list(map(float,[1,0,0,0]*4))
    assert seqs_encoded[2]==list(map(float,[0,1,0,0]*4))
    assert seqs_encoded[3]==list(map(float,[0,0,1,0]*4))
    assert seqs_encoded[4]==list(map(float,[0,0,0,1]*4))

    # pass