
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, \
    precision_recall_curve, average_precision_score, roc_auc_score, \
    roc_curve

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# customised classes
from claim_dataset import *
from claim_net import *
from part2_claim_classifier import *

def ClaimClassifierHyperParameterSearch(x_train, y_train, x_val, y_val):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    grid = {"learning_rate" : 0,
            "neuron_num" : 0,
            "batch_size" : 0,
            "over" : 0,
            "roc_auc" : 0}
    for neuron_num in range(6, 54, 9):
        for learning_rate in np.arange(1e-4, 2e-3, 2e-4): # 1-e3 is the default lr for adam
            for over in np.arange(0.2, 0.45, 0.05):
                for batch_size in range(24, 64, 8):
                    print("learning_rate: {}, neuron_num: {}, batch_size: {}, over : {}"\
                          .format(learning_rate, neuron_num, batch_size, over))

                    # Oversampling
                    oversampling = SMOTE(over) # ratio from 1 to 0 after oversampling
                    x_tra, y_tra = oversampling.fit_resample(x_train, y_train)
                    under = RandomUnderSampler(0.9) # ratio from 1 to 0 after undersampling
                    x_tra, y_tra = under.fit_resample(x_tra, y_tra)
                    x_tra = np.array(x_tra)
                    y_tra = np.array(y_tra).reshape(len(y_tra), 1)

                    # Create a network
                    claim_classifier = ClaimClassifier(input_dim = 9, 
                                                    output_dim = 1, 
                                                    neurons = [neuron_num, neuron_num, neuron_num], 
                                                    activations = ["relu", "sigmoid"], 
                                                    loss_func = "bce", 
                                                    optimiser = "adam", 
                                                    learning_rate = learning_rate, 
                                                    max_epoch = 100, 
                                                    batch_size = batch_size)

                    # Train the network
                    loss_hist, loss_val_hist, roc_auc_hist = \
                        claim_classifier.fit(x_tra, y_tra, x_val, y_val, True)

                    #Predict
                    prob_val = claim_classifier.predict(x_val)

                    # Evaluation
                    roc_auc = roc_auc_score(y_val, prob_val)
                    if roc_auc > grid["roc_auc"]:
                        grid["roc_auc"] = roc_auc
                        grid["learning_rate"] = learning_rate
                        grid["neuron_num"] = neuron_num
                        grid["batch_size"] = batch_size
                        grid["over"] = over
                        print(grid)
                    else: 
                        print(roc_auc)

    return grid

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
    hyper_main()