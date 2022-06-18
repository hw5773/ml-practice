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

def run(fname):
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(",")
            if not tmp[0].isnumeric():
                print ("not number: {}".format(int(tmp[0])))

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", metavar="<input dataset>", help="Input Dataset File Path", type=str, required=True)
    parser.add_argument("-o", "--output", metavar="<output dataset>", help="Output Dataset File Path", type=str)
    parser.add_argument("-l", "--log", metavar="<log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)>", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")
    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.input):
        logging.error("Input File ({}) does not exist".format(args.input))
        sys.exit(1)

    run(args.input)

if __name__ == "__main__":
	main()
