from random import random 
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from math import log
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def crossEntropyLoss(Y_predict, Y_init, margin=1e-20):
    np.clip(Y_predict, margin, 1 - margin, out=Y_predict)
    log_pred = np.vectorize(log)(Y_predict)
    log_pred_comp = np.vectorize(log)(1 - Y_predict)
    product = np.multiply(Y_init, log_pred)
    return (-product.sum(axis=0)).mean()

def sigmoid (x): return (1/(1 + np.exp(-x)))      # activation function
def sigmoid_(x): return (x * (1 - x))             # derivative of sigmoid
def softmax(Z):
    expZ = np.exp(Z)
    result = []
    for z in expZ:
        d = []
        d.append(z[0] / (z[0] + z[1]))
        d.append(z[1] / (z[0] + z[1]))
        result.append(d)
    return np.array(result)

def predict(X):
    Y_predict = []
    for x in X:
        if x[0] > x[1]:
            Y_predict.append(1)
        else:
            Y_predict.append(0)
    return (np.array(Y_predict))

def scale_data(df):
    return (df - df.mean()) / df.std()

class Neuron(object): 
    def __init__(self, prev_layer_size):
        self.weights = []
        for i in range(prev_layer_size):
            np.random.seed()
            self.weights.append(np.random.uniform())
    def __repr__(self):
        return '{"bias":"' + str(self.bias) + '" , "output":"' + str(self.output) + '" , "weights":' + str(self.weights) +  '}' 

class Layer(object):
    neurons = []
    outputs = []
    weights = []
    def __init__(self, nb_neurons, prev_nb_neurons, activation): 
        self.activation = activation
        self.neurons = [Neuron(prev_nb_neurons) for i in range(nb_neurons)]
    
    def add_weights(self):
        self.weights = [neuron.weights for neuron in self.neurons]
        self.weights = np.array(self.weights)
    
    def add_bias(self):
        # self.bias = np.random.rand(1, len(self.neurons))
        self.bias = np.zeros((1, len(self.neurons)))
        self.bias = np.array(self.bias)

    def add_outputs(self):
        self.outputs = [neuron.output for neuron in self.neurons]
        self.outputs = np.array(self.outputs)

    def __repr__(self): 
        return '{"activation":"'+str(self.activation)+'" , "neurons":'+str(self.neurons)+' , "outputs":'+str(self.outputs)+'}' 

class Network(object): 
    layers = [] 
    def add_layer(self, nb_neurons, activation='identity'): 
        layer = Layer(nb_neurons,
                      len(self.layers[-1:][0].neurons) if len(self.layers) > 0 else 0, activation) 
        self.layers.append(layer) 
    def __repr__(self): 
        return '{"Network":{"layers":' + str(self.layers) + '}}'

np.seterr(all = 'ignore')
# read dataset
columns = []
for i in range(32):
    columns.append('column_' + str(i))
df = pd.read_csv('data.csv', names=columns)
# numerize column_1
df['M'] = df['column_1'].map({'M': 1, 'B': 0})
df['B'] = df['column_1'].map({'M': 0, 'B': 1})
# split dataset
target = df['M']
# create training and testing vars
X_train_0, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
mean, std = X_train_0.mean().tolist(), X_train_0.std().tolist()
X_train_0 = X_train_0.drop('column_1', axis=1)
X_train = X_train_0.drop('M', axis=1)
X_test = X_test.drop('B', axis=1)
X_test = X_test.drop('M', axis=1)
X_test.to_csv('test.csv', index=False)
Y_M = np.reshape(y_train.values, (X_train.shape[0], 1))
Y_B = np.reshape(X_train.B.values, (X_train.shape[0], 1))
X_train = X_train.drop('B', axis=1)
X_train_scale = scale_data(X_train)
# Network
epochs = 100    # Number of iterations
inputLayerSize, hiddenLayerSize_1, hiddenLayerSize_2, outputLayerSize = X_train.shape[1], 16, 8,2
L = 0.01  # learning rate
network = Network()
network.add_layer(inputLayerSize, 'input')
network.add_layer(hiddenLayerSize_1, 'hidden_1')
network.add_layer(hiddenLayerSize_2, 'hidden_2')
network.add_layer(outputLayerSize, 'output')
# Initialize Input layer
X = np.reshape(X_train_scale.values, (X_train.shape[0], X_train.shape[1]))
Y = np.concatenate((Y_M, Y_B), axis=1)
for index, neuron in enumerate(network.layers[0].neurons):
    neuron.output = X[index]
network.layers[0].add_outputs()
network.layers[1].add_weights()
network.layers[2].add_weights()
network.layers[3].add_weights()
network.layers[1].add_bias()
network.layers[2].add_bias()
network.layers[3].add_bias()
#### Shuffle
X, Y = shuffle(X, Y, random_state=np.random.RandomState())
Y_init = Y[:, [0]]
for i in range(epochs):
    ############# feedforward
    ### Hidden layer 1 ###
    z_h_1 = X.dot(network.layers[1].weights.T) + network.layers[1].bias
    network.layers[1].outputs = sigmoid(z_h_1)
    ### Hidden layer 2 ###
    z_h_2 = network.layers[1].outputs.dot(network.layers[2].weights.T) + network.layers[2].bias
    network.layers[2].outputs = sigmoid(z_h_2)
    ###output###
    z_o = np.dot(network.layers[2].outputs, network.layers[3].weights.T) + network.layers[3].bias
    network.layers[3].outputs = softmax(z_o)
    ### Backpropagation
    ## Output layer
    delta_z_o = network.layers[3].outputs - Y
    delta_w13 = network.layers[2].outputs
    dw_o = np.dot(delta_z_o.T, delta_w13) / X.shape[0]
    db_o = np.sum(delta_z_o, axis=0, keepdims=True) / X.shape[0]
    ## Hidden layer 2
    delta_a_h_2 = np.dot(delta_z_o, network.layers[3].weights)
    delta_z_h_2 = sigmoid_(network.layers[2].outputs)
    d = delta_a_h_2 * delta_z_h_2
    delta_w12 = network.layers[1].outputs
    dw_h_2 = np.dot(d.T, delta_w12) / X.shape[0]
    db_h_2 = np.sum(d, axis=0, keepdims=True) / X.shape[0]
    ## Hidden layer 1
    delta_a_h_1 = delta_z_h_2.dot(network.layers[2].weights)
    delta_z_h_1 = sigmoid_(network.layers[1].outputs)
    d =  delta_a_h_1 * delta_z_h_1
    delta_w11 = X
    dw_h_1 = np.dot(d.T, delta_w11) / X.shape[0]
    db_h_1 = np.sum(d, axis=0, keepdims=True) / X.shape[0]
    # print(db_h_1)
    ### Update weights and bias
    network.layers[1].weights -= L * dw_h_1
    network.layers[2].weights -= L * dw_h_2
    network.layers[3].weights -= L * dw_o
    network.layers[1].bias -= L * db_h_1
    network.layers[2].bias -= L * db_h_2
    network.layers[3].bias -= L * db_o
    #### Loss function
    Y_predict = predict(network.layers[3].outputs)
    ### Cross entropy
    loss = -np.mean(Y_init * np.log(Y_predict.T + 1e-9))
    print("epoch {}/{} - loss: {} - val_loss: {}".format(i + 1, epochs, loss, 0))

# print(Y_predict)
# accuracy = (Y_init == Y_predict).mean()
# print(accuracy * 100)
print('{:.2f}'.format(accuracy_score(Y_init, Y_predict)))
weights_dic = {}
weights_dic['hidden_1_w'] = network.layers[1].weights
weights_dic['hidden_2_w'] = network.layers[2].weights
weights_dic['output_w'] = network.layers[3].weights
weights_dic['hidden_1_b'] = network.layers[1].bias
weights_dic['hidden_2_b'] = network.layers[2].bias
weights_dic['output_b'] = network.layers[3].bias
weights_dic['mean'] = mean
weights_dic['std'] = std
np.save('weights.npy', weights_dic)
