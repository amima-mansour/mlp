import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def standardize(vector, mean, std):
    return (vector - mean) / std
def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=1, keepdims=True)

np.seterr(all = 'ignore')
f = np.load('weights.npy', allow_pickle=True)
hidden_layer_1 = f.item().get('hidden_1')
hidden_layer_2 = f.item().get('hidden_2')
output_layer = f.item().get('output')
mean = f.item().get('mean')
std = f.item().get('std')
df = pd.read_csv('test.csv')
Y = df['column_1'].map({'M': 1, 'B': 0}).values
df = df.drop('column_1', axis=1)
for i in range(32):
    if i != 1:
        df['column_' + str(i)] = standardize(df['column_' + str(i)], mean[i], std[i])
X = df.values
add = np.ones((X.shape[0], 1))
X = np.concatenate((X, add), axis=1)
### hidden layer 1 ###
z_h_1 = np.dot(X, hidden_layer_1)
hidden_outputs_1 = sigmoid(z_h_1)
### hidden layer 2 ###
z_h_2 = np.dot(hidden_outputs_1, hidden_layer_2)
hidden_outputs_2 = sigmoid(z_h_2)
### output layer ###
z_o = np.dot(hidden_outputs_2, output_layer.T)
final_outputs = softmax(z_o)
Y_predict = []
for x in final_outputs:
    if x[0] > x[1]:
        Y_predict.append(1)
    else:
        Y_predict.append(0)
Y_predict = np.array(Y_predict)
print(accuracy_score(Y, Y_predict))