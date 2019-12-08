import numpy as np


def sigmoid (x): return (1/(1 + np.exp(-x)))

def softmax(Z):
    expZ = np.exp(Z)
    result = []
    for z in expZ:
        d = []
        d.append(z[0] / (z[0] + z[1]))
        d.append(z[1] / (z[0] + z[1]))
        result.append(d)
    return np.array(result)

def final_outputs(X):
    Y_predict = []
    for x in X:
        if x[0] > x[1]:
            Y_predict.append(1)
        else:
            Y_predict.append(0)
    return (np.array(Y_predict))

def predict(X, weights_hidden_1, bias_hidden_1, weights_hidden_2, bias_hidden_2, weights_outputs, bias_output):
    ### hidden layer 1 ###
    z_h_1 = np.dot(X, weights_hidden_1.T) + bias_hidden_1
    hidden_outputs_1 = sigmoid(z_h_1)
    ### hidden layer 2 ###
    z_h_2 = np.dot(hidden_outputs_1, weights_hidden_2.T) + bias_hidden_2
    hidden_outputs_2 = sigmoid(z_h_2)
    ### output layer ###
    z_o = np.dot(hidden_outputs_2, weights_outputs.T) + bias_output
    outputs = softmax(z_o)
    return (final_outputs(outputs))