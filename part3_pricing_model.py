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
        X_clean = [] # stores processed training dataset     
        NUM = []
        ORD = []
        CAT = []
        VH = []
        LOC_CODE = []
        # Merge vehicle make & model
        make = X_raw[:,Data.vh_make.value]
        model = X_raw[:,Data.vh_model.value]
        for i in range(len(make)):
            VH.append(make[i] + model[i])
        CAT.append(np.array(VH))
        
        # Merge location codes 
        # TODO: remove, produces 16028 combinations
        loc_code = np.array(X_raw[:,COMMUNE_CANTON_DIST_REG[0].value]).reshape(-1,1)
        for i in range(1, (len(COMMUNE_CANTON_DIST_REG))):
            index = COMMUNE_CANTON_DIST_REG[i].value
            new_col = np.array(X_raw[:,index]).reshape(-1,1)
            loc_code = np.hstack((loc_code, new_col))
        
        for row in range(len(loc_code)):
            code = ''
            for data in loc_code[row]:
                code += str(data)
            LOC_CODE.append(code)
        CAT.append(np.array(LOC_CODE))

        # Split attributes according to data type
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

        # Filter data that has #_of_nan >= threshold
        threshold = 3
        rm_rows = []
        for row in range(len(NUM)):
            count = 0
            for data in NUM[row]:
                if math.isnan(data):
                    count += 1
            if count >= threshold:
                rm_rows.append(row)

        NUM = np.delete(NUM, rm_rows, 0)
        ORD = np.delete(ORD, rm_rows, 0)
        CAT = np.delete(CAT, rm_rows, 0)

        # Fill in missing values
        imp_NA = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="NA") 
        imp_replace_nan = SimpleImputer(missing_values=np.nan, strategy="mean")     
        imp_replace_zero = SimpleImputer(missing_values=0, strategy="mean")   

        # for ORDINAL type, replace nan occurrences with "NA"
        ORD = imp_NA.fit_transform(ORD)

        # for NUMERICAL type, replace nan and 0 occurrences with mean
        NUM = imp_replace_nan.fit_transform(NUM)
        NUM = imp_replace_zero.fit_transform(NUM)

        # for CATEGORICAL type, check if values are string/numeric
        CAT = imp_NA.fit_transform(CAT)

        oe = preprocessing.OrdinalEncoder(categories=[['Mini','Median1','Median2','Maxi'],['Retired','WorkPrivate','Professional','AllTrips']])

        # Transform categorical strings into binary labels
        lb = preprocessing.LabelBinarizer()
        for i in range (CAT.shape[1]):  
            labels = lb.fit_transform(CAT[:,i])
            if labels.shape[1] == 1:
                CAT[:,i] = labels.flatten()
            else: 
                print(CAT[0,i], labels.shape)
                #  add arrays of labels into position?
                for row in range(labels.shape[0]):
                    CAT[row][i] = labels[row].flatten()
                
        ORD = oe.fit_transform(ORD)

        
        # Normalise with MinMaxScaler
        # scaler = preprocessing.MinMaxScaler()
        # scaler.fit(X_clean)

        # return np.asarray(scaler.transform(X_clean))
        # return X_clean # YOUR CLEAN DATA AS A NUMPY ARRAY

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
    # headers = ["pol_bonus","pol_coverage","pol_duration","pol_sit_duration","pol_payd","pol_usage","pol_insee_code","drv_age1","drv_age2",
    # "drv_sex1","drv_sex2","drv_age_lic1","drv_age_lic2","vh_age","vh_cyl","vh_din","vh_fuel","vh_make","vh_model","vh_sale_begin","vh_sale_end","vh_speed","vh_type","vh_value","vh_weight","town_mean_altitude","town_surface_area","population","commune_code","canton_code","city_district_code","regional_department_code","made_claim"]

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
