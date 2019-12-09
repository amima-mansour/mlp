import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tools import predict, sigmoid
from math import log
import sys
import argparse

def standardize(vector, mean, std):
    return (vector - mean) / std

def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=1, keepdims=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Set Dataset to predict")
parser.add_argument("--weights", help="Set weights file")
args = parser.parse_args()
np.seterr(all = 'ignore')
try:
    f = np.load(args.weights, allow_pickle=True)
    weights_hidden_1 = f.item().get('hidden_1_w')
    weights_hidden_2 = f.item().get('hidden_2_w')
    weights_outputs = f.item().get('output_w')
    bias_hidden_1 = f.item().get('hidden_1_b')
    bias_hidden_2 = f.item().get('hidden_2_b')
    bias_output = f.item().get('output_b')
    mean = f.item().get('mean')
    std = f.item().get('std')
    df = pd.read_csv(args.dataset)
    j = 0
    for i in range(30):
        if i != 1:
            df['column_' + str(i)] = standardize(df['column_' + str(i)], mean[j], std[j])
            j += 1
    Y_M = df['column_1'].map({'M': 1, 'B': 0}).values
    Y_B = df['column_1'].map({'M': 0, 'B': 1}).values
    Y_M = np.reshape(Y_M, (Y_B.shape[0], 1))
    Y_B = np.reshape(Y_B, (Y_B.shape[0], 1))
    Y = np.concatenate((Y_M, Y_B), axis=1)
    df = df.drop('column_1', axis=1)
    X = df.values
    Y_predict = predict(X, weights_hidden_1, bias_hidden_1, weights_hidden_2, bias_hidden_2, weights_outputs, bias_output)
    print('Accuracy test = {:.2f}'.format(accuracy_score(Y[:, 0], Y_predict)))
    ### Cross entropy
    cross = -np.sum(Y[:, 0] * np.log(Y_predict.T + 1e-9) + (1 - Y[:, 0]) * np.log(1 - Y_predict.T + 1e-9)) / Y_predict.shape[0]
    print('Cross Entropy value = {:.5f}'.format(cross))
except:
    print("Error dataset to predict or weights")