import numpy as np
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

class ClaimClassifier():

    def __init__(self, input_dim, neurons, activations, train):
        """
        input_dim {int} -- Dimension of input (excluding batch dimension).
        neurons {list} -- Number of neurons in each layer represented as a
            list (the length of the list determines the number of layers).
        activations {list} -- List of the activation function to use for
            each layer.
        """
        self._layers = []
        self.input_dim = input_dim
        self.train_config = train

        n_inputs = input_dim
        for i in range(len(neurons)):
            self._layers.append(nn.Linear(n_inputs, neurons[i]))
            if activations[i] == "relu":
                self._layers.append(nn.ReLU())
            elif activations[i] == "sigmoid":
                self._layers.append(nn.Sigmoid())
            elif activations[i] == "softmax":
                self._layers.append(nn.Softmax(dim=1))
            elif activations[i] == "tanh":
                self._layers.append(nn.Tanh())
            n_inputs = neurons[i]  
        
        if activations[-1] == "sigmoid":
            self.threshold = 0.5
        elif activations[-1] == "tanh":
            self.threshold = 0
        else:
            print("Only sigmoid and tanh is acceptable as an activation function of the output layer.")
        
        self.model = nn.Sequential(*self._layers)
        
    def forward(self, x):
        return self.model(x)
        

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
        # DECISION: which norm
        return normalize(X_raw, norm='max', axis=0) # YOUR CLEAN DATA AS A NUMPY ARRAY
    
    def calc_loss(self, prediction, annotation, loss_fun):
        if loss_fun == "bce":
            return F.binary_cross_entropy(prediction, annotation) 
        if loss_fun == "mse":
            return F.mse_loss(prediction, annotation)
        if loss_fun == "cross_entropy":
            return F.cross_entropy(prediction, annotation)
        return F.mse_loss(prediction, annotation)

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
        X_clean = self._preprocessor(X_raw)
        # training configs
        learning_rate = self.train_config["learning_rate"]
        loss_fun = self.train_config["loss_fun"]
        n_epochs = self.train_config["nb_epoch"]
        batch_size = self.train_config["batch_size"]
        shuffle = False
        if self.train_config["shuffle_flag"]:
            shuffle = True
        
        # Prepare dataset for training
        targets = np.array([t for t in y_raw])
        dataset = np.append(X_clean, targets, axis=1)
        optimiser = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            mini_batches = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)
            for data in mini_batches:
                X = data[:, :self.input_dim]
                y = data[:, self.input_dim:]
                self.model.zero_grad()
                y_pred = self.forward(X.float())
                loss = self.calc_loss(y_pred, y.float(), loss_fun)
                loss.backward()
                optimiser.step()

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
        X_clean = self._preprocessor(X_raw)
        y_pred = self.model(torch.from_numpy(X_clean).float())
        return binary_conv(y_pred, self.threshold) # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self, prediction, annotation):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        return accuracy_score(prediction, annotation)
        

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

def binary_conv(predictions, split_value):
    predictions_binary = []

    for i in range (len(predictions)):
        if (predictions[i] < split_value):
            predictions_binary.append(0)
        else:
            predictions_binary.append(1)
    return np.asarray(predictions_binary)


def main():
    input_dim = 9
    # hidden_layers = 2
    # Params for layers:
    neurons = [10, 10, 1] 
    activations = ["relu", "relu", "sigmoid"]

    # Params for training:
    train = { 
        "batch_size": 8,
        "nb_epoch": 2,
        "learning_rate": 0.01,
        "loss_fun": "bce",
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

    # net._preprocessor(x_train)
    # claim_classifier = ClaimClassifier(hidden_layers)

    net.fit(x_train, y_train)
    
    # prediction_train = claim_classifier.register_parameter(x_train)
    y_pred = net.predict(x_test)
    print(net.evaluate_architecture(y_pred, y_test))
    # TODO: Evaluation of prediction_train and prediction_test

if __name__ == "__main__":
    main()
