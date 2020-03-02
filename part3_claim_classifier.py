import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, \
    precision_recall_curve, average_precision_score, roc_auc_score, \
    roc_curve

import matplotlib.pyplot as plt

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler

# customised classes
from claim_dataset import *
from claim_net import *

class BaseClassifier():

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
        self._net_aid = ClaimNet(input_dim, output_dim, neurons, activations)
        self._net_aid2 = ClaimNet(input_dim, output_dim, neurons, activations)
        # print("=== The created network is: ===")
        # print(self._net)
        # print()
        self._max_epoch = max_epoch
        self._scaler = None
        self._batch_size = batch_size
        self.classes_ = [0, 1]

        if loss_func == "bce":
            self._loss_func = nn.BCELoss() 
        elif loss_func == "mse":
            self._loss_func = nn.MSELoss()
        elif loss_func == "cross_entropy":
            self._loss_func = nn.CrossEntropyLoss()

        # print("=== The parameters are : ===")
        # for name, param in self._net.named_parameters():
        #     print(name, param.data)
        # print()
        if optimiser == "sgd":
            self._optimiser = optim.SGD(self._net.parameters(), learning_rate, weight_decay = 0.01)
            self._optimiser_aid = optim.SGD(self._net_aid.parameters(), learning_rate, weight_decay = 0.01)
            self._optimiser_aid2 = optim.SGD(self._net_aid2.parameters(), learning_rate, weight_decay = 0.01)
        elif optimiser == "adam":
            self._optimiser = optim.Adam(self._net.parameters(), learning_rate)
            self._optimiser_aid = optim.Adam(self._net_aid.parameters(), learning_rate)


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
            # self._scaler = MinMaxScaler()
            self._scaler = StandardScaler()
            self._scaler.fit(X_raw)
        X_raw = self._scaler.transform(X_raw)

        return np.array(X_raw)

    def fit(self, X_raw, y_raw, X_val = None, y_val = None, early_stop = False):
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
        validation = ClaimDataset(self._preprocessor(X_val), y_val)

        # Training
        ap_hist = []
        roc_auc_hist = []
        loss_hist = []
        loss_val_hist = []
        for e in range(self._max_epoch):
            # print("* Epoch: ", e)
            # Update
            losses = []
            dataset_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
            for x_batch, y_batch in dataset_loader:
                # Forward
                self._net.zero_grad()
                output = self._net(x_batch)

                # Loss
                loss = self._loss_func(output, y_batch)
                losses.append(loss.item())

                # Backward
                loss.backward()

                # Optimise
                self._optimiser.step()
            
            # Average loss
            average_loss = sum(losses)/len(losses)
            loss_hist.append(average_loss)
            # print("   Loss: ", average_loss)

            # Evaluate
            validation_loader = DataLoader(validation, batch_size=len(X_val))
            for x_validation, y_validation in validation_loader:

                prediction = self._net(x_validation)

                loss = self._loss_func(prediction, y_validation)
                val_loss = loss.item()
                loss_val_hist.append(val_loss)

                average_precision = average_precision_score(y_val, prediction.cpu().detach().numpy())
                ap_hist.append(average_precision)

                roc_auc = roc_auc_score(y_val, prediction.cpu().detach().numpy())
                roc_auc_hist.append(roc_auc)

                # print("   AUC:  ", roc_auc)
                # print("   AP:   ", average_precision)
                # print("   Loss: ", val_loss)

            # Early stopping
            if e > 20 and early_stop:
                if (((roc_auc_hist[-1] - roc_auc_hist[-2]) + \
                    (roc_auc_hist[-2] - roc_auc_hist[-3]) + \
                    (roc_auc_hist[-3] - roc_auc_hist[-4])) < 0):
                        # print("Early stopping ...")
                        break

        # Train again
        roc_auc_hist_aid = []
        loss_hist_aid = []
        for e in range(self._max_epoch):
            # print("model2 * Epoch: ", e)
            # Update
            losses_aid = []
            dataset_loader_aid = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
            for x_batch, y_batch in dataset_loader_aid:
                # Forward
                self._net_aid.zero_grad()
                output_aid = self._net_aid(x_batch)

                # Loss
                loss_aid = self._loss_func(output_aid, y_batch)
                losses_aid.append(loss_aid.item())

                # Backward
                loss_aid.backward()

                # Optimise
                self._optimiser_aid.step()
            
            # Average loss
            average_loss_aid = sum(losses_aid)/len(losses_aid)
            loss_hist_aid.append(average_loss_aid)
            # print("   Loss: ", average_loss_aid)

            # Evaluate
            validation_loader_aid = DataLoader(validation, batch_size=len(X_val))
            for x_validation, y_validation in validation_loader_aid:

                prediction = self._net_aid(x_validation)

                roc_auc = roc_auc_score(y_val, prediction.cpu().detach().numpy())
                roc_auc_hist_aid.append(roc_auc)

                # print("   AUC:  ", roc_auc)

            # Early stopping
            if e > 20 and early_stop:
                if (((roc_auc_hist_aid[-1] - roc_auc_hist_aid[-2]) + \
                    (roc_auc_hist_aid[-2] - roc_auc_hist_aid[-3]) + \
                    (roc_auc_hist_aid[-3] - roc_auc_hist_aid[-4])) < 0):
                        # print("Early stopping ...")
                        break

        # Train again
        roc_auc_hist_aid2 = []
        loss_hist_aid2 = []
        for e in range(self._max_epoch):
            # print("model3 * Epoch: ", e)
            # Update
            losses_aid2 = []
            dataset_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
            for x_batch, y_batch in dataset_loader:
                # Forward
                self._net_aid2.zero_grad()
                output_aid2 = self._net_aid2(x_batch)

                # Loss
                loss_aid2 = self._loss_func(output_aid2, y_batch)
                losses_aid2.append(loss_aid2.item())

                # Backward
                loss_aid2.backward()

                # Optimise
                self._optimiser_aid2.step()
            
            # Average loss
            average_loss_aid2 = sum(losses_aid2)/len(losses_aid2)
            loss_hist_aid2.append(average_loss_aid2)
            # print("   Loss: ", average_loss_aid2)

            # Evaluate
            validation_loader = DataLoader(validation, batch_size=len(X_val))
            for x_validation, y_validation in validation_loader:

                prediction = self._net_aid2(x_validation)

                roc_auc = roc_auc_score(y_val, prediction.cpu().detach().numpy())
                roc_auc_hist_aid2.append(roc_auc)

                # print("   AUC:  ", roc_auc)

            # Early stopping
            if e > 20 and early_stop:
                if (((roc_auc_hist_aid2[-1] - roc_auc_hist_aid2[-2]) + \
                    (roc_auc_hist_aid2[-2] - roc_auc_hist_aid2[-3]) + \
                    (roc_auc_hist_aid2[-3] - roc_auc_hist_aid2[-4])) < 0):
                        # print("Early stopping ...")
                        break

        return loss_hist, loss_val_hist, roc_auc_hist, self


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
        predictions_aid = self._net_aid(X_clean)
        predictions_aid2 = self._net_aid2(X_clean)
        predictions = (predictions + predictions_aid + predictions_aid2) / 3

        return np.asarray(torch.Tensor.cpu(predictions).detach().numpy())

    def predict_proba(self, X_raw):
        proba = self.predict(X_raw)
        return np.append(np.subtract(np.ones((len(proba), 1)), proba), proba, axis=1)

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
    plt.figure(figsize=(6, 18))

    plt.subplot(311)
    plt.step(recall, precision)
    plt.title('2-class Precision-Recall curve for make_claim = 1: AP={0:0.4f}'.format(ap), fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)

    plt.subplot(312)
    plt.plot(fpr, tpr)
    plt.title('2-class ROC for make_claim = 1: AUC={0:0.4f}'.format(auc), fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

    plt.subplot(313)
    plt.hist(probability, bins=80)
    plt.title('Distribution of Positive Probability', fontsize=18)
    plt.xlabel('Probability', fontsize=16)
    plt.ylabel('Portion', fontsize=16)

    plt.show()
