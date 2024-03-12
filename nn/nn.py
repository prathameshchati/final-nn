# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    # for nn_arch, it was originally Union(int, str) - this kept raising an error
    # https://stackoverflow.com/questions/38854282/do-union-types-actually-exist-in-python
    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]], 
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # each row in the W_curr matrix represents the neuron in the next layer, the number columns indicate the weights of each neuron in the previous activation layer (A_prev); the biases (b_curr) are added
        # the output Z_curr is of dimensions output_dim x batches
        Z_curr=np.dot(W_curr, A_prev)+b_curr

        # pass through activation function
        if (activation=='sigmoid'):
            A_curr=self._sigmoid(Z_curr)
        elif (activation=='relu'):
            A_curr=self._relu(Z_curr)

        return A_curr, Z_curr
        # pass

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
            # define cache as param_dict as we are going to update it
        cache={}

        # set A_curr as the input batches but transpose to have the batches along the columns and features along the rows; this matches it to the W matrix, which contains the weights for the current layer in each row
        A_curr=X.T
        cache['A0']=A_curr # add as initial layer

        # go through each layer and call _single_forward
        for idx, layer in enumerate(self.arch):
            A_prev=A_curr
            layer_idx=idx+1
            A_curr, Z_curr=self._single_forward(self._param_dict['W' + str(layer_idx)], self._param_dict['b' + str(layer_idx)], A_prev, layer['activation'])

            # add the A_curr and Z_curr to cache (on the last run, we get the final output layer)
            cache['A' + str(layer_idx)]=A_curr # has dimensions output_dim x batch_size
            cache['Z' + str(layer_idx)]=Z_curr # has dimensions output_dim x batch_size

        # transpose A_curr to have batches along the rows and feature salong the columns - this was raising some errors
        return A_curr.T, cache
        # pass

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        # https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76
        # dA_curr is the dC/dA_last, which is multiplied in the backpropagation step with dA/dZ, the output of this multiplication is dZ_curr, which represents dC/dZ -> we compute this first
        # dA_curr has dimensions output_dim x batch_size, as does Z_curr and dZ_curr
        if (activation_curr=='sigmoid'):
            dZ_curr=self._sigmoid_backprop(dA_curr, Z_curr)
        elif (activation_curr=='relu'):
            dZ_curr=self._relu_backprop(dA_curr, Z_curr)

        # dA_prev, represents dC/dA_prev, can be thought of as expanding the chain rule using dZcurr (dC/dZ) and dZ/dA_prev; the derivative of Z_curr with respect to A_prev is simply W_curr
        # this means, taking the dot product of dZ_curr and W_curr will give us dA_prev
        dA_prev=np.dot(W_curr.T, dZ_curr)

        # dZ/dW, the derivative of the linear combination with respect to weights, is simply A_prev, which has dimensions of previous layer, i.e. input_layer x batch_size
        # W_curr, has dimensions output_dim x input_dim (A_prev length); so dW_curr must also be this dimension, and we can take the dot product of dZ_curr and A_prev.T
        # Note, dW_curr represents dC/dW, which is the full chain multiplication of the subcomponents
        dW_curr=np.dot(dZ_curr, A_prev.T)/dZ_curr.shape[1] # normalize by number of samples in batch (batch_size)

        # db_curr corresponds to dC/db, the derivative of cost wrt b, which is similar to dW curr but now we have to consider the term we multiply, which was A_prev for dW_curr. 
        # taking the derivative of Z wrt b gives you 1, so we get dZ_curr again where the dimensions are output_dim x batch_size, however, we only have one set of b values per layer.
        # thus, we average across the batch for dZ_curr
        db_curr=np.mean(dZ_curr, axis=1)
        return dA_prev, dW_curr, db_curr
        # pass

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76
        # first compute dA_curr given the loss_function, y, and y_hat
        if (self._loss_func=="bce"): # binary cross entropy
            dA_curr=self._binary_cross_entropy_backprop(y, y_hat)
        elif (self._loss_func=="mse"): # mean squared error
            dA_curr=self._mean_squared_error_backprop(y, y_hat)


        # initialize gradient dictionary
        grad_dict={}
        
        # the cache and param_dict is indexed such that the input layer is A0 and the final layer is AN where N is the number of layers. So with a three layer system, we would have A0, A1, and A2; The weights and biases are labeled for A1 and A2.
        # iterate and enumerate the list in reverse to get the components that are fed into  
        # https://stackoverflow.com/questions/529424/traverse-a-list-in-reverse-order-in-python
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx=idx+1
            dA_prev, dW_curr, db_curr=self._single_backprop(self._param_dict['W'+str(layer_idx)], self._param_dict['b'+str(layer_idx)], cache['Z'+str(layer_idx)], cache['A'+str(idx)], dA_curr, layer['activation'])
            dA_curr=dA_prev # update dA_curr as the previous dA since we are going backward

            # update gradient dicts with the same labels as the param_dict
            grad_dict['W'+str(layer_idx)]=dW_curr
            grad_dict['b'+str(layer_idx)]=db_curr

        return grad_dict
        # pass

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # iterate through layers and update the weights by subtracting the gradient for the given layer times the learning rate
        for idx, layer in enumerate(self.arch):
            layer_idx=idx+1
            self._param_dict['W'+str(layer_idx)]-=self._lr*grad_dict['W'+str(layer_idx)]

            # reshape the b update to 2D
            update_b=self._lr*grad_dict['b'+str(layer_idx)]
            update_b=update_b.reshape(len(update_b), 1)
            self._param_dict['b'+str(layer_idx)]-=update_b
        # pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # intialize loss lists
        per_epoch_loss_train=[]
        per_epoch_loss_val=[]

        print("X_train dim: ", X_train.shape)
        print("y_train dim: ", y_train.shape)
        print("X_val dim: ", X_val.shape)
        print("y_val dim: ", y_val.shape)

        # split training data into batches # https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
        X_train_batches=[X_train[i:i + self._batch_size] for i in range(0, len(X_train), self._batch_size)]  
        y_train_batches=[y_train[i:i + self._batch_size] for i in range(0, len(y_train), self._batch_size)]  

        # iterate and train wrt number of epoch
        for e in range(self._epochs):
            # all of the training is done with the batches iteratively and each time the loss is stored in the loss_train_list, which is averaged to give you the loss_train for the epoch
            loss_train_list=[]
            for X_train_batch, y_train_batch in zip(X_train_batches, y_train_batches):
                # first train via forward alg. and get training loss
                y_hat_train, cache_train=self.forward(X_train_batch)
            
                if (self._loss_func=="bce"): 
                    loss_train=self._binary_cross_entropy(y_train_batch, y_hat_train)
                elif (self._loss_func=="mse"): 
                    loss_train=self._mean_squared_error(y_train_batch, y_hat_train)

                loss_train_list.append(loss_train)

                # update weights via backprop
                grad_dict=self.backprop(y_train_batch, y_hat_train, cache_train)
                self._update_params(grad_dict)

            # weighted average of the training losses where the weights are the length of each batch. If the batches are balanced in size, then the weighting does nothing and it is simply equal to the unweighted average
            loss_train=(np.sum(np.array(loss_train_list)*np.array([len(b) for b in X_train_batches])))/X_train.shape[0]
            per_epoch_loss_train.append(loss_train) # store training loss

            # run validation on val data and store validation loss
            y_hat_val, cache_val=self.forward(X_val)
            # y_hat_val=self.predict(X_val)
            if (self._loss_func=="bce"): 
                loss_val=self._binary_cross_entropy(y_val, y_hat_val)
            elif (self._loss_func=="mse"): 
                loss_val=self._mean_squared_error(y_val, y_hat_val)
            per_epoch_loss_val.append(loss_val) # store validation loss

        return per_epoch_loss_train, per_epoch_loss_val
        # pass

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """

        # call the forward alg
        y_hat, cache=self.forward(X)
        return y_hat
        # pass

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1/(1+np.exp(-Z))
        # pass

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # the derivative of the sigmoid function is f(Z)(1-f(Z)) where f is the sigmoid 
        dZ=self._sigmoid(Z)*(1-self._sigmoid(Z))

        # multiple dZ with dA to get our dC/dZ - derivative of the cost function with respect to Z
        dZ=dA*dZ # changed to dA.T to fix broadcasting error

        return dZ
        # pass

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # https://www.digitalocean.com/community/tutorials/relu-function-in-python
        # https://stackoverflow.com/questions/32322281/numpy-matrix-binarization-using-only-one-expression
        return np.maximum(0.0, Z) # streamlined from np.array([np.maximum(0,z) for z in Z.T])
        # pass

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # the derivative of relu returns 0 if the value is below or equal to 0 or 1 otherwise; convert relu to binary
        dZ=self._relu(Z)
        dZ=np.where(Z>0, 1, 0) # [0 if z<=0 else 1 for z in Z] was causing errors
        dZ=dA*dZ # multiply dA
        return dZ
        # pass

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # bianry cross entropy from hw7-regression
        loss=(-1/len(y))*(np.transpose(y)@np.log(y_hat)+np.transpose(1-y)@np.log(1-y_hat))
        return loss
        # pass

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # https://towardsdatascience.com/nothing-but-numpy-understanding-creating-binary-classification-neural-networks-with-e746423c8d5c
        # bce gradient should be different from the regression derivative, which is with respect to weight 
        dA=-(y/y_hat)+((1-y)/(1-y_hat))
        return dA
        # pass

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # mse is just the mean of the differences between y and y_hat squared 
        loss=np.mean((y-y_hat)**2)
        return loss
        # pass

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # the derivative is 2 times the difference between y and y_hat
        dA=-(2/len(y))*(y-y_hat)
        return dA
        # pass