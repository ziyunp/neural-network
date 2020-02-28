from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch.nn as nn
from data_processing import *
import math

def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
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
        self.base_classifier = None # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        # Group data into 3 types and process separately
        NUM = []
        ORD = []
        CAT = []

        # Filter attributes that have # of nan or zeros > 10% of data points
        threshold = 0.1
        rm_attr = []
        for feat in range(X_raw.shape[1]):
            count = 0
            for data in X_raw[:,feat]:
                if not data or data != data:
                    count += 1
            if count > threshold * X_raw.shape[0]:
                rm_attr.append(feat)
        for att in rm_attr:
            if att in [e.value for e in ORDINAL]:
                ORDINAL.remove(Data(att))
            if att in [e.value for e in NUMERICAL]:
                NUMERICAL.remove(Data(att))
            if att in [e.value for e in CATEGORICAL]:
                CATEGORICAL.remove(Data(att))

        # Remove rows that have # of nan or zeros > 10% of #_of_features
        rm_rows = []
        for row in range(len(X_raw)):
            count = 0
            for i in range(X_raw.shape[1]):
                if i not in rm_attr:
                    data = X_raw[row][i]
                    if not data or data != data:
                        count += 1
            if count > threshold * X_raw.shape[1]:
                rm_rows.append(row)
        X_raw = np.delete(X_raw, rm_rows, 0)

        # Merge vehicle make & model
        VH = []
        if Data.vh_make in CATEGORICAL and Data.vh_model in CATEGORICAL:
            make = X_raw[:,Data.vh_make.value]
            model = X_raw[:,Data.vh_model.value]
            for i in range(len(make)):
                VH.append(make[i] + model[i])
            CAT.append(np.array(VH))
            CATEGORICAL.remove(Data.vh_make)
            CATEGORICAL.remove(Data.vh_model)

        # Convert location codes from float -> string
        if Data.commune_code in CATEGORICAL:
            CATEGORICAL.remove(Data.commune_code)
            commune = X_raw[:,Data.commune_code.value]
            commune_code = [str(c) for c in commune]
            CAT.append(np.array(commune_code))
        if Data.canton_code in CATEGORICAL:
            CATEGORICAL.remove(Data.canton_code)
            canton = X_raw[:,Data.canton_code.value]
            canton_code = [str(c) for c in canton]
            CAT.append(np.array(canton_code))
        if Data.city_district_code in CATEGORICAL:
            CATEGORICAL.remove(Data.city_district_code)
            city_district = X_raw[:,Data.city_district_code.value]
            city_district_code = [str(c) for c in city_district]
            CAT.append(np.array(city_district_code))

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
        imp_NA = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="NA") 
        imp_replace_nan = SimpleImputer(missing_values=np.nan, strategy="mean")     
        imp_replace_zero = SimpleImputer(missing_values=0, strategy="mean")   

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
        
        # Normalise with MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X_clean)

        return np.asarray(scaler.transform(X_clean)) # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw, claims_raw):
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

        # TODO: Check input data dimensions
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
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
        # X_clean = self._preprocessor(X_raw)


        return  # return probabilities for the positive class (label 1)

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


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

def main():
    input_dim = 35
    # # hidden_layers = 2
    # print("Number of input variables: " , input_dim)

    dataset = pd.read_csv('part3_training_data.csv')  
    # np.random.shuffle(dataset)

    x = dataset.iloc[:,:input_dim]
    y = dataset.iloc[:,input_dim+1:] # not including claim_amount
    model = PricingModel()
    x_prep = model._preprocessor(x.to_numpy())
    # test if arrays can be passed to linear layer as training data
    # linear_layer = nn.Linear(input_dim, 1)
    # linear_layer(x_prep)
    # split_idx_train = int(0.6 * len(x))
    # split_idx_val = int((0.6 + 0.2) * len(x))

    # x_train = x[:split_idx_train]
    # y_train = y[:split_idx_train]
    # x_val = x[split_idx_train:split_idx_val]
    # y_val = y[split_idx_train:split_idx_val]
    # x_test = x[split_idx_val:]
    # y_test = y[split_idx_val:]

if __name__ == "__main__":
    main()
