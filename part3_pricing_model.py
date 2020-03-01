from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from data_processing import *
from part3_helper import *
from part3_claim_classifier import *

def fit_and_calibrate_classifier(classifier, X, y, x_val = None, y_val = None, early_stop = None):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train, x_val, y_val, early_stop)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, input_dim, output_dim, neurons, activations, loss_fun, optimiser, learning_rate, epoch, batch_size, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self._rm_attr = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = ClaimClassifier(input_dim, output_dim, neurons, activations, loss_fun, optimiser, learning_rate, epoch, batch_size)


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw, train=False):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        # TODO:
        # function for report analysis
        # deal with outliers in NUM
        # note that this will be called in prediction - add param to flag train/predict

        # Group data into 3 types and process separately
        NUM = []
        ORD = []
        CAT = []
        
        # TODO: Store removed attributes and apply on prediction set
        # Filter attributes that have # of nan or zeros > 10% of data points
        THRESHOLD = 0.1

        if train:
            # If training, save attributes to remove, else, use the stored list
            self._rm_attr = filter_attributes(X_raw, THRESHOLD)
            # Store removed rows that have # of nan or zeros > 10% of #_of_features and apply to y_train
            self._rm_rows = filter_data(X_raw, THRESHOLD, self._rm_attr)
            X_raw = np.delete(X_raw, self._rm_rows, 0)
    
        for att in self._rm_attr:
            if att in [e.value for e in ORDINAL]:
                ORDINAL.remove(Data(att))
            if att in [e.value for e in NUMERICAL]:
                NUMERICAL.remove(Data(att))
            if att in [e.value for e in CATEGORICAL]:
                CATEGORICAL.remove(Data(att))


        # Group attributes according to data type
        for i in range(len(NUMERICAL)):
            index = NUMERICAL[i].value
            NUM.append(X_raw[:,index])
        for j in range(len(ORDINAL)):
            index = ORDINAL[j].value
            ORD.append(X_raw[:,index])
        for k in range(len(CATEGORICAL)):
            index = CATEGORICAL[k].value
            CAT.append(X_raw[:,index])

        NUM = np.array(NUM).transpose()
        ORD = np.array(ORD).transpose()
        CAT = np.array(CAT).transpose()

        # Fill in missing values
        # TODO: use IterativeImputer?
        imp_NA = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="NA") 
        imp_replace_nan = IterativeImputer(random_state=0, missing_values=np.nan)     
        imp_replace_zero = IterativeImputer(random_state=0, missing_values=0)   

        # for ORDINAL type, replace nan with "NA"
        ORD = imp_NA.fit_transform(ORD)

        # for NUMERICAL type, replace nan and 0 with mean
        NUM = imp_replace_nan.fit_transform(NUM)
        NUM = imp_replace_zero.fit_transform(NUM)

        # for CATEGORICAL type, replace nan with "NA"
        CAT = imp_NA.fit_transform(CAT)

        # Transform ORDINAL strings into numerical labels
        oe = preprocessing.OrdinalEncoder(categories=[['Mini','Median1','Median2','Maxi'],['Retired','WorkPrivate','Professional','AllTrips']])
        ORD = oe.fit_transform(ORD)

        # Merge ORD and NUM into X_clean
        X_clean = np.hstack((ORD, NUM))
        # Transform CATEGORICAL values into binary labels
        lb = preprocessing.LabelBinarizer()
        for i in range (CAT.shape[1]):  
            sparse_matrix = lb.fit_transform(CAT[:,i])
            X_clean = np.hstack((X_clean, sparse_matrix))    

        # Use normalisation in base_classifier
        return X_clean.astype(dtype = 'float32') # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw, claims_raw, x_val = None, y_val = None, early_stop = None):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """

        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw, True)
        y_clean = np.delete(y_raw, self._rm_rows, 0)

        x_val_clean = self._preprocessor(x_val)
        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_clean, x_val_clean, y_val, early_stop)
        else:
            self.base_classifier.fit(X_clean, y_clean, x_val_clean, y_val, early_stop)
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        y_pred = self.base_classifier.predict(X_clean)

        return  y_pred # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)

    def evaluate_architecture(self, probability, annotation):
        self.base_classifier.evaluate_architecture(probability, annotation)

def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

def over_sampling(dataset, ratio):
    """Performs oversampling to the given dataset according to ratio 
    Parameters
    ----------
    dataset : raw dataset with attributes appended with 1 label 
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

def main():
    input_dim = 35 # num of attributes of interest
    # # hidden_layers = 2
    # print("Number of input variables: " , input_dim)

    dataset = pd.read_csv('part3_training_data.csv')  
    # np.random.shuffle(dataset)

    x = dataset.iloc[:,:input_dim].to_numpy()
    y = dataset.iloc[:,input_dim+1:].to_numpy() # not including claim_amount
    claim_amount = dataset.iloc[:, input_dim].to_numpy()
    claim_amount = np.array([float(c) for c in claim_amount])
    split_idx_train = int(0.8 * len(dataset))
    split_idx_val = int((0.8 + 0.1) * len(dataset))

    x_train = x[:split_idx_train]
    y_train = y[:split_idx_train]
    x_val = x[split_idx_train:split_idx_val]
    y_val = y[split_idx_train:split_idx_val]
    x_test = x[split_idx_val:]
    y_test = y[split_idx_val:]
   # Oversampling
    train = np.append(x_train, y_train, 1)
    train = over_sampling(train, 1)
    np.random.shuffle(train)
    x_train = train[:, :input_dim]
    y_train = train[:, input_dim:]
    y_train = y_train.astype(dtype = 'float32')

    input_dim = 25 # num of attributes after cleaning
    output_dim = 1
    neurons = [16, 20, 24, 28, 32]
    activations = ["relu", "sigmoid"]
    loss_fun = "bce"
    optimiser = "sgd"
    learning_rate = 0.5e-4
    epoch = 100
    batch_size = 200

    model = PricingModel(input_dim, output_dim, neurons, activations, loss_fun, optimiser, learning_rate, epoch, batch_size)

    # Train the network
    model.fit(x_train, y_train, claim_amount, x_val, y_val, False)
    
#     claim_classifier.save_model()

    #Predict
    prob_train = model.predict_claim_probability(x_train)
    # Evaluation
    print()
    print("------- The result of training set is: ------")
    model.evaluate_architecture(prob_train, y_train)

    #Predict for validation
    prob_val = model.predict_claim_probability(x_val)

    # Evaluation for validation
    print()
    print("------- The result of validation set is: ------")
    model.evaluate_architecture(prob_val, y_val)

#    plot_precision_recall(prob_val, y_val)
    #Predict for validation
    prob_test = model.predict_claim_probability(x_test)

    # Evaluation for test
    print()
    print("------- The result of validation set is: ------")
    model.evaluate_architecture(prob_test, y_test)

    premium = model.predict_premium(x_test)
    print("premium: ", premium)


if __name__ == "__main__":
   main()
