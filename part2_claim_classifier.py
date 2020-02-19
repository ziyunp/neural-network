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

    def __init__(self, input_dim, neurons, activations, loss_func, optimiser, learning_rate, epoch, batch_size):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self._net = ClaimNet(input_dim, neurons, activations)
        print("=== The created network is: ===")
        print(self._net)
        print()
        self._epoch = epoch
        self._normaliser = None
        self._batch_size = batch_size

        if loss_func == "bce":
            self._loss_func = nn.BCELoss() 
        elif loss_func == "mse":
            self._loss_func = nn.MSELoss()
        elif loss_func == "cross_entropy":
            self._loss_func = nn.CrossEntropyLoss()

        print("=== The parameters are : ===")
        for name, param in self._net.named_parameters():
            print(name, param.data)
        print()
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
            self._normaliser = Normalizer(norm='l1')
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
        dataset_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Training
        pri = False
        for _ in range(self._epoch):
            for x_batch, y_batch in dataset_loader:
                # Forward
                output = self._net(x_batch)

                # Loss
                loss = self._loss_func(output, y_batch)
                if not pri:
                    print("=== Print out the backprop orders: ===")
                    print("The 1st should be Loss function:   ", loss.grad_fn)
                    print("The 2nd should be the sigmoid:     ", loss.grad_fn.next_functions[0][0])
                    print("The 3rd should be the liner layer: ", loss.grad_fn.next_functions[0][0].next_functions[0][0])
                    print("The 4th should be the sigmoid:     ", loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
                    print()
                    pri = True

                # Backprop
                print("=== Gradient change: ===")
                print("ll1.bias.grad before backward: ") 
                print(self._net._ll1.bias.grad)
                self._net.zero_grad()
                loss.backward()
                print("ll1.bias.grad after backward: ") 
                print(self._net._ll1.bias.grad)
                print()

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
        
        # Produce raw predictions
        predictions = self._net(X_clean)

        # Convert to binary classes
        predictions_binary = []
        for i in range (len(predictions)):
            if (predictions[i] < 0.5):
                predictions_binary.append(0)
            else:
                predictions_binary.append(1)
        return np.asarray(predictions_binary)


    def evaluate_architecture(self, prediction, annotation):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        
        print("=== Confusion Matrix ===")
        print(confusion_matrix(annotation, prediction))

        print("=== Classification Report ===")
        print(classification_report(annotation, prediction))


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
    neurons = [4, 1]
    activations = ["sigmoid", "sigmoid"]
    loss_fun = "bce"
    optimiser = "sgd"
    learning_rate = 0.01
    epoch = 700
    batch_size = 10
    claim_classifier = ClaimClassifier(input_dim, neurons, activations, loss_fun, optimiser, learning_rate, epoch, batch_size)

    # Train the network
    claim_classifier.fit(x_train, y_train)

    #Predict
    prediction_val = claim_classifier.predict(x_val)
    # prediction_test = claim_classifier.predict(x_test)
    
    claim_classifier.evaluate_architecture(prediction_val.squeeze(), y_val)

if __name__ == "__main__":
    main()
