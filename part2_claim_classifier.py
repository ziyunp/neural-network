import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import normalize, Normalizer
from sklearn.metrics import classification_report, confusion_matrix

# customised classes
from claim_dataset import *
from claim_net import *

class ClaimClassifier():

    def __init__(self, input_dim, neurons, activations, loss_func, optimiser, learning_rate, epoch):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self._net = ClaimNet(input_dim, neurons, activations)
        self._epoch = epoch
        self._normaliser = None

        if loss_func == "bce":
            self._loss_func = nn.BCELoss() 
        elif loss_func == "mse":
            self._loss_func = nn.MSELoss()
        elif loss_func == "cross_entropy":
            self._loss_func = nn.CrossEntropyLoss()

        if optimiser == "sgd":
            self._optimiser = optim.SGD(self._net.parameters(), learning_rate)


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
        X_raw = np.transpose(X_raw)
        if self._normaliser == None:
            self._normaliser = Normalizer(norm='max')
            self._normaliser.fit(X_raw)
        X_raw = self._normaliser.transform(X_raw)

        return np.transpose(X_raw)

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
        # Create a dataset loader
        dataset = ClaimDataset(self._preprocessor(X_raw), y_raw)
        dataset_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Training
        for _ in range(self._epoch):
            for x_batch, y_batch in dataset_loader:
                # Forward
                output = self._net(x_batch)

                # Loss
                loss = self._loss_func(output, y_batch)

                # Backprop
                self._net.zero_grad()
                loss.backward()

                # Optimise
                self._optimiser.step()
        
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
        X_clean = torch.from_numpy(self._preprocessor(X_raw)).float()

        return self._net(x_batch)

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        
        # print("=== Performance of the model on the training data ===")
        # print(confusion_matrix(self.y_train, self.predict_train))
        # print(classification_report(self.y_train, self.predict_train))

        # print("=== Performance of the model on the test data ===")
        # print(confusion_matrix(self.y_test, self.predict_test))
        # print(classification_report(self.y_test, self.predict_test))

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
    
    # Read the dataset
    dataset = np.genfromtxt('part2_training_data.csv', delimiter=',', skip_header=1)
    np.random.shuffle(dataset)

    x = dataset[:, :9]
    y = dataset[:, 10:] # not including claim_amount 

    split_idx_train = int(0.6 * len(dataset))
    split_idx_val = int((0.6 + 0.2) * len(dataset))

    x_train = x[:split_idx_train]
    y_train = y[:split_idx_train]
    x_val = x[split_idx_train:split_idx_val]
    y_val = y[split_idx_train:split_idx_val]
    x_test = x[split_idx_val:]
    y_test = y[split_idx_val:]

    # Create a network
    input_dim = 9
    neurons = [18, 1]
    activations = ["relu", "sigmoid"]
    loss_fun = "bce"
    optimiser = "sgd"
    learning_rate = 0.01
    epoch = 2
    claim_classifier = ClaimClassifier(input_dim, neurons, activations, loss_fun, optimiser, learning_rate, epoch)

    # Train the network
    claim_classifier.fit(x_train, y_train)

    # prediction_train = claim_classifier.register_parameter(x_train)
    # prediction_test = claim_classifier.predict(x_test)
    
    # TODO: Evaluation of prediction_train and prediction_test

if __name__ == "__main__":
    main()
