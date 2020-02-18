import numpy as np
import pickle

from nn_lib import *

import torch
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix

class ClaimClassifier(torch.nn.Module):

    def __init__(self, hidden_size):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.batch_size = 0
        self.input_size = 0
        self.hidden_size = hidden_size
        self.output_size = 1

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

        self.batch_size = X_raw.shape[0]
        self.input_size = X_raw.shape[1]

        preprocessor = Preprocessor(X_raw)

        return preprocessor.apply(X_raw)

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

        # if (not y_raw):
        #     print("y_raw not provided")
        #     return

        
        X_clean = torch.from_numpy(self._preprocessor(X_raw))
        y_raw = torch.from_numpy(y_raw)

        # print(X_clean)

        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.output_size),
        )

        loss_fn = torch.nn.MSELoss(reduction='sum')

        learning_rate = 1e-4

        for t in range(500):
            # Forward pass
            y_pred = model(X_clean.float()) # X_clean needs to be a tensor

            print(y_pred)
            print(y_raw)


            loss = loss_fn(y_pred, y_raw.float())
            if t % 100 == 99:
                print(t, loss.item())

            # Zero the gradients before running the backward pass.
            model.zero_grad()

            # Backward pass
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

        self.model = model

        return model
        
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
        X_clean = torch.from_numpy(self._preprocessor(X_raw))

        # YOUR CODE HERE
        y_predict = model(X_clean)
 
        return y_predict

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
    input_dim = 9
    hidden_layers = 2
    
    # drv_age1, vh_age, vh_cyl, vh_din, pol_bonus, vh_sale_begin, vh_sale_end, 
    # vh_value, vh_speed, claim_amount, made_claim
    dataset = np.genfromtxt('part2_training_data.csv',delimiter=',',skip_header=1)
    np.random.shuffle(dataset)

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

    # claim_classifier = ClaimClassifier(hidden_layers)

    # claim_classifier.fit(x_train, y_train)

    # prediction_train = claim_classifier.register_parameter(x_train)
    # prediction_test = claim_classifier.predict(x_test)
    
    # TODO: Evaluation of prediction_train and prediction_test

if __name__ == "__main__":
    main()
