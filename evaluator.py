import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MaxAbsScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SOURCE_IP_ADDR_IDX=3
DESTINATION_IP_ADDR_IDX=5

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_dataset(fname):
    dataframe = pandas.read_csv(fname, engine='python')
    dataset = dataframe.values
    logging.debug("dataset: {}".format(dataset))

    X = dataset[:, :-3]
    y = dataset[:, -3]

    logging.debug("X: {}".format(X))
    logging.debug("y: {}".format(y))

    y = y.reshape((len(y), 1))
    y = preprocessing_label(y)

    return X, y
    
def preprocessing(X, y):
    X = preprocessing_data(X)
    y = preprocessing_label(y)
    return X, y

def preprocessing_data(X):
    for i in range(len(X)):
        if "0x" in X[i][3]:
            X[i][3] = str(int(X[i][3], 16))

        if "0x" in X[i][5]:
            X[i][5] = str(int(X[i][5], 16))
    
    ct = ColumnTransformer(transformers=[("ohe", OneHotEncoder(), [1, 2, 4])], remainder='passthrough')
    X = ct.fit_transform(X)

    scaler = MaxAbsScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X

def preprocessing_label(y):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    logging.debug("y: {}".format(y))
    return y

def make_pipeline(X_train, y_train):
    numerical_idxes = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_idxes = X_train.select_dtypes(include=['str', 'object', 'bool']).columns

    logging.info("numerical_idxes: {}".format(numerical_idxes))
    logging.info("categorical_idxes: {}".format(categorical_idxes))

    t = [('cat', OneHotEncoder(), categorical_idxes), ('num', MinMaxScaler(), numerical_idxes)]
    ct = ColumnTransformer(transformers=t)

    model = SVR(kernel='rbf', gamma='scale', C=100)
    pipeline = Pipeline(steps=[('prep', ct), ('m', model)])

    return pipeline

def find_best_models(pipeline, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=5)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=tscv, return_estimator=True)

    flst = scores["test_f1_macro"]
    mvalue = max(flst)

    idxes = []
    for idx in range(len(flst)):
        if lst[idx] == mvalue:
            idxes.append(idx)

    candidates = []
    models = scores["estimator"]

    for idx in idxes:
        m = models[idx]

        included = True
        for c in candidates:
            if m.__class__.__name__ == c.__class__.__name__:
                mparams = m.get_params()
                cparams = c.get_params()

                for k in mparams.keys():
                    if k not in cparams:
                        included = False
                        break
                    elif mparams[k] != cparams[k]:
                        included = False
                        break

        if not included or len(candidates) == 0:
            candidates.append(m)

    return candidates

def evaluate_models(models, X_test, y_test):
    report = {}
    idx = 0
    for model in models:
        y_pred = model.predict(X_test)
        ret = {}
        ret["model"] = model
        ret["accuracy"] = accuracy_score(y_test, y_pred)
        ret["precision"] = precision_score(y_test, y_pred)
        ret["recall"] = recall_score(y_test, y_pred)
        ret["f1_score"] = f1_score(y_test, y_pred)
        report[idx] = ret
        idx += 1
    return report

def report_results(reports):
    print (">>>>> Reports: {} <<<<<".format(len(reports)))

    idx = 0
    for k in reports:
        print ("===== Model {} =====".format(idx))
        print ("Model: {}".format(reports[k]["model"].__class__.__name__))
        print ("Parameters: {}".format(reports[k]["model"].get_params()))
        print ("Accuracy: {}".format(reports[k]["accuracy"]))
        print ("Precision: {}".format(reports[k]["precision"]))
        print ("Recall: {}".format(reports[k]["recall"]))
        print ("F1-score: {}".format(reports[k]["f1_score"]))
        print ("====================")
        idx += 1

def run(train_name, test_name):
    logging.info("Before loading the training dataset")
    X_train, y_train = load_dataset(train_name)
    logging.info("After loading the dataset")

    logging.info("Before making the pipeline")
    pipeline = make_pipeline(X_train, y_train)
    logging.info("After making the pipline")

    logging.info("Before finding the best models")
    models = find_best_models(pipeline, X_train, y_train)
    logging.info("After finding the best models")

    logging.info("Before loading the test dataset")
    X_test, y_test = load_dataset(test_name)
    logging.info("After loading the test dataset")

    logging.info("Before evaluating the models")
    reports = evaluate_models(models, X_test, y_test)
    logging.info("After evaluating the models")

    logging.info("Before reporting the results")
    report_results(reports)
    logging.info("After reporting the results")

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", metavar="<training dataset>", help="Training Dataset File Path", type=str, required=True)
    parser.add_argument("-u", "--test", metavar="<test dataset>", help="Test Dataset File Path", type=str, required=True)
    parser.add_argument("-l", "--log", metavar="<log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)>", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")
    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("Training File ({}) does not exist".format(args.input))
        sys.exit(1)

    if not os.path.exists(args.test):
        logging.error("Test File ({}) does not exist".format(args.test))
        sys.exit(1)

    run(args.training, args.test)

if __name__ == "__main__":
	main()
