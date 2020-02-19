import numpy as np
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.preprocessing import normalize

class ClaimClassifier():

    def __init__(self, input_dim, neurons, activations, train):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        """
        input_dim {int} -- Dimension of input (excluding batch dimension).
        neurons {list} -- Number of neurons in each layer represented as a
            list (the length of the list determines the number of layers).
        activations {list} -- List of the activation function to use for
            each layer.
        """
        self._layers = []
        n_inputs = input_dim
        for i in range(len(neurons)):
            self._layers.append(nn.Linear(n_inputs, neurons[i]))
            if activations[i] == "relu":
                self._layers.append(nn.ReLU())
            elif activations[i] == "sigmoid":
                self._layers.append(nn.Sigmoid())
            elif activations[i] == "softmax":
                self._layers.append(nn.Softmax())
            elif activations[i] == "tanh":
                self._layers.append(nn.Tanh())
            n_inputs = neurons[i]  

        self.train_config = train
        self._model = nn.Sequential(*self._layers)
        
    def forward(self, x):
        # move to function that calls forward
        x = torch.Tensor(x)
        y = self._model(x)
        return y
        

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        return normalize(X_raw, axis=0)
        # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE
        pass

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters

def main():
    input_dim = 9
    # hidden_layers = 2
    # Params for layers:
    neurons = [10, 10, 1] 
    activations = ["relu", "relu", "sigmoid"]

    # Params for training:
    train = { 
        "batch_size": 8,
        "nb_epoch": 1000,
        "learning_rate": 0.01,
        "shuffle_flag": True
    }

    net = ClaimClassifier(input_dim, neurons, activations, train)

    dataset = np.genfromtxt('part2_training_data.csv',delimiter=',',skip_header=1)
    np.random.shuffle(dataset)

    # drv_age1, vh_age, vh_cyl, vh_din, pol_bonus, vh_sale_begin, vh_sale_end, 
    # vh_value, vh_speed, claim_amount, made_claim

    x = dataset[:, :input_dim]
    y = dataset[:, input_dim+1:] # not including claim_amount 

    split_idx_train = int(0.6 * len(x))
    split_idx_val = int((0.6 + 0.2) * len(x))

    x_train = x[:split_idx_train]
    y_train = y[:split_idx_train]
    x_val = x[split_idx_train:split_idx_val]
    y_val = y[split_idx_train:split_idx_val]
    x_test = x[split_idx_val:]
    y_test = y[split_idx_val:]

    net._preprocessor(x_train)
    # claim_classifier = ClaimClassifier(hidden_layers)

    # claim_classifier.fit(x_train, y_train)

    # prediction_train = claim_classifier.register_parameter(x_train)
    # prediction_test = claim_classifier.predict(x_test)
    
    # TODO: Evaluation of prediction_train and prediction_test

if __name__ == "__main__":
    main()
