import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def standardize(vector, mean, std):
    return (vector - mean) / std
def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid

f = np.load('weights.npy', allow_pickle=True)
hidden_layer = f.item().get('hidden_1')
output_layer = f.item().get('output')
mean = f.item().get('mean')
std = f.item().get('std')
df = pd.read_csv('test.csv')
Y = df.column_1
for i in range(32):
    df['column_' + str(i)] = standardize(df['column_' + str(i)], mean[i], std[i])
X = df.drop('column_1', axis=1).values
add = np.ones((X.shape[0], 1))
X = np.concatenate((X, add), axis=1)
### first layer ###
z_h = np.dot(X, hidden_layer)
hidden_outputs = sigmoid(z_h)
z_o = np.dot(hidden_outputs, output_layer)
final_outputs = sigmoid(z_o)
Y_predict = []
for x in final_outputs:
    if x >= 0.5:
        Y_predict.append(1)
    else:
        Y_predict.append(0)

print(accuracy_score(Y, Y_predict))
