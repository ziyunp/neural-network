import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, \
    precision_recall_curve, average_precision_score, roc_auc_score, \
    roc_curve

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

# customised classes
from claim_net import *

class ClaimClassifier():

    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 neurons, 
                 activations, 
                 loss_func, 
                 optimiser, 
                 learning_rate, 
                 max_epoch, 
                 batch_size):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self._net = ClaimNet(input_dim, output_dim, neurons, activations)
        print("=== The created network is: ===")
        print(self._net)
        print()
        self._max_epoch = max_epoch
        self._scaler = None
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
        elif optimiser == "adam":
            self._optimiser = optim.Adam(self._net.parameters(), learning_rate)


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
        if self._scaler == None:
            self._scaler = MinMaxScaler()
            self._scaler.fit(X_raw)
        X_raw = self._scaler.transform(X_raw)

        return np.array(X_raw)

    def fit(self, X_raw, y_raw, X_val = None, y_val = None, early_stop = None):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable
        X_val : ndarray (optional)
            An array, this is the validation data as downloaded
        y_val : ndarray (optional)
            A one dimensional array, this is the binary target variable
        early_stop : int (optional)
            The limit where the early stop should happen

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """
        # Create a dataset loader
        dataset = ClaimDataset(self._preprocessor(X_raw), y_raw)
        
        # Training
        ap_hist = []
        roc_auc_hist = []
        loss_hist = []
        for e in range(self._max_epoch):
            print("* Epoch: ", e)
            # Update
            losses = []
            dataset_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
            for x_batch, y_batch in dataset_loader:
                # Forward
                output = self._net(x_batch)

                # Loss
                loss = self._loss_func(output, y_batch)
                losses.append(loss.item())

                # Backward
                self._net.zero_grad()
                loss.backward()

                # Optimise
                self._optimiser.step()
            
            # Average loss
            average_loss = sum(losses)/len(losses)
            loss_hist.append(average_loss)
            print("   Loss: ", average_loss)

            # Evaluate
            prediction = self.predict(X_val)
            average_precision = average_precision_score(y_val, prediction)
            ap_hist.append(average_precision)
            roc_auc = roc_auc_score(y_val, prediction)
            roc_auc_hist.append(roc_auc)
            print("   AUC:  ", roc_auc)
            print("   AP:   ", average_precision)

            # Early stopping
            if e > 2:
                # if (abs(ap_hist[-1] - ap_hist[-2]) + \
                #     abs(ap_hist[-2] - ap_hist[-3])) / 2 < early_stop:
                #         print("Early stopping ...")
                #         break
                if (abs(roc_auc_hist[-1] - roc_auc_hist[-2]) + \
                    abs(roc_auc_hist[-2] - roc_auc_hist[-3])) / 2 < early_stop:
                        print("Early stopping ...")
                        break


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

        return np.asarray(torch.Tensor.cpu(predictions).detach().numpy())

    def convert_to_binary(self, predictions, threshold = 0.5):
        """Convert to binary classes
        """
        predictions_binary = []
        for i in range (len(predictions)):
            if (predictions[i] < threshold):
                predictions_binary.append(0)
            else:
                predictions_binary.append(1)
        return np.asarray(predictions_binary)

    def evaluate_architecture(self, probability, annotation):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        Paramters
        ---------
        probability : ndarray
        annotation : ndarray
        """
        # Convert the predicted probability to predicion
        prediction = self.convert_to_binary(probability)

        print("=== Confusion Matrix ===")
        print(confusion_matrix(annotation, prediction))
        print()

        print("=== Classification Report ===")
        print(classification_report(annotation, prediction))
        print()

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)

    def set_epoch(self, epoch):
        self._epoch = epoch

def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(x_train, y_train, x_val, y_val):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    grid = {"learning_rate" : 1e-3,
            "neuron_num" : 6,
            "batch_size" : 16,
            "over" : 0.7,
            "roc_auc" : 0}
    for learning_rate in np.arange(0.00002, 0.0005, 0.00002): # 1-e3 is the default lr for adam
        for neuron_num in range(6, 18, 2):
            for batch_size in range(16, 64, 8):
                for over in np.arange(0.7, 1, 0.1):
                    print("learning_rate: {}, neuron_num: {}, batch_size: {}, over : {}"\
                          .format(learning_rate, neuron_num, batch_size, over))

                    # sampling
                    oversampling = SMOTE(over)
                    x, y = oversampling.fit_resample(x_train, y_train)
                    x = np.array(x)
                    y = np.array(y).reshape(len(y), 1)

                    # Create network
                    claim_classifier = ClaimClassifier(input_dim = 9, 
                                                    output_dim = 1, 
                                                    neurons = [neuron_num, neuron_num, neuron_num, neuron_num], 
                                                    activations = ["relu", "sigmoid"], 
                                                    loss_func = "bce", 
                                                    optimiser = "adam", 
                                                    learning_rate = learning_rate, 
                                                    max_epoch = 100, 
                                                    batch_size = batch_size)

                    # Train the network
                    claim_classifier.fit(x, y, x_val, y_val, 0.00008)

                    #Predict
                    prob_train = claim_classifier.predict(x_val)

                    # Evaluation
                    roc_auc = roc_auc_score(y_val, prob_train)
                    if roc_auc > grid["roc_auc"]:
                        grid["roc_auc"] = roc_auc
                        grid["learning_rate"] = learning_rate
                        grid["neuron_num"] = neuron_num
                        grid["batch_size"] = batch_size
                        grid["over"] = over
                        print(grid)

    return grid

def over_sampling(dataset, ratio):
    """Performs oversampling to the given dataset according to ratio 
    Parameters
    ----------
    dataset : raw dataset with 9 attributes appended with 1 label 
    ratio : a float from 0 to 1, any number larger then 1 will be treated as 1,
            smaller will be treated as 0
            make_claim (label 1) to not_make_claim (label 0)

    Returns
    -------
    ndarray : Dataset after being oversampled
    """
    label1 = []
    label0 = []
    for data in dataset:
        if data[-1] == 1:
            label1.append(data)
        else:
            label0.append(data)
    if ratio < 0:
        ratio = 0
    elif ratio > 1:
        ratio = 1
    current_ratio = len(label1) / len(label0)
    for _ in range(int(ratio / current_ratio)):
        label0 = np.append(label0, label1, 0)
        
    return label0

def plot_precision_recall(probability, annotation):
    """Plot precisin-recall curve

    Parameters
    ----------
    probability : ndarray, the probability of the label being 1
    annotation: ndarray, the actual labels
    """
    precision, recall, thresholds = \
        precision_recall_curve(annotation, probability, pos_label=1)
    ap = average_precision_score(annotation, probability, pos_label=1)

    fpr, tpr, thresholds_roc = roc_curve(annotation, probability, pos_label=1)
    auc = roc_auc_score(annotation, probability)
    plt.figure(figsize=(10, 18))

    plt.subplot(311)
    plt.step(recall, precision)
    plt.title('2-class Precision-Recall curve for make_claim: AP={0:0.4f}'.format(ap), fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)

    plt.subplot(312)
    plt.step(tpr, fpr)
    plt.title('2-class ROC for make_claim: AUC={0:0.4f}'.format(auc), fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)

    plt.subplot(313)
    plt.hist(probability, bins=40)
    plt.title('Distribution of Positive Probability', fontsize=18)
    plt.xlabel('Probability', fontsize=16)
    plt.ylabel('Portion', fontsize=16)

    plt.show()

def main():
    
    # Read the dataset
    dataset = np.genfromtxt('part2_training_data.csv', delimiter=',', skip_header=1)
    np.random.shuffle(dataset)

    x = dataset[:, :9]
    y = dataset[:, 10:] # not including claim_amount 

    split_idx_train = int(0.8 * len(dataset))
    split_idx_val = int((0.8 + 0.1) * len(dataset))

    x_val = x[split_idx_train:split_idx_val]
    y_val = y[split_idx_train:split_idx_val]
    x_test = x[split_idx_val:]
    y_test = y[split_idx_val:]

    x_train = x[:split_idx_train]
    y_train = y[:split_idx_train]

    # Oversampling
    train = np.append(x_train, y_train, 1)
    train = over_sampling(train, 1)
    np.random.shuffle(train)
    x_train = train[:, :9]
    y_train = train[:, 9:]

    # Create a network
    claim_classifier = None
    # claim_classifier = load_model()
    if claim_classifier == None:
        input_dim = 9
        output_dim = 1
        neurons = [6, 6, 6, 6, 6]
        activations = ["relu", "sigmoid"]
        loss_fun = "bce"
        optimiser = "sgd"
        learning_rate = 1e-3
        epoch = 1
        batch_size = 4
        claim_classifier = ClaimClassifier(input_dim, output_dim, neurons, activations, loss_fun, optimiser, learning_rate, epoch, batch_size)
    # claim_classifier.set_epoch(2)

    recalls = []
    for i in range(30): 
        # Train the network
        claim_classifier.fit(x_train, y_train, x_val, y_val, 0.0002)
        claim_classifier.save_model()

        #Predict
        prob_train = claim_classifier.predict(x_train)
    
        # Evaluation
        print()
        print("------- The result of ", i, "is: ------")
        claim_classifier.evaluate_architecture(prob_train, y_train)

    #Predict for validation
    prob_val = claim_classifier.predict(x_val)
    # prediction_test = claim_classifier.predict(x_test)

    # Evaluation for validation
    print()
    print("------- The result of validation set is: ------")
    claim_classifier.evaluate_architecture(prob_val, y_val)
    # claim_classifier.evaluate_architecture(prediction_test.squeeze(), y_test)

    plot_precision_recall(prob_val, y_val)

def hyper_main():
    # Read the dataset
    dataset = np.genfromtxt('part2_training_data.csv', delimiter=',', skip_header=1)
    np.random.shuffle(dataset)

    x = dataset[:, :9]
    y = dataset[:, 10:] # not including claim_amount 

    split_idx_train = int(0.7 * len(dataset))
    split_idx_val = int((0.7 + 0.15) * len(dataset))

    x_val = x[split_idx_train:split_idx_val]
    y_val = y[split_idx_train:split_idx_val]
    x_test = x[split_idx_val:]
    y_test = y[split_idx_val:]

    x_train = x[:split_idx_train]
    y_train = y[:split_idx_train]

    ClaimClassifierHyperParameterSearch(x_train, y_train, x_val, y_val)

if __name__ == "__main__":
    # main()
    hyper_main()
