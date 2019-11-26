from random import random 
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from math import *
from sklearn.metrics import accuracy_score

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
def softmax(x): return np.exp(x) / sum(np.exp(x))
def scale_data(df):
    return (df - df.mean()) / df.std()

class Neuron(object): 
    bias = 0 
    output = 0 
    weights = [] 
    def __init__(self, prev_layer_size): 
        self.weights = [random() for i in range(prev_layer_size)]
    def __repr__(self): 
        return '{"bias":"' + str(self.bias) + '" , "output":"' + str(self.output) + '" , "weights":' + str(self.weights) +  '}' 

class Layer(object): 
    activation = '' 
    neurons = [] 
    outputs = []
    weights = []
    def __init__(self, nb_neurons, prev_nb_neurons, activation): 
        self.activation = activation
        self.neurons = [Neuron(prev_nb_neurons) for i in range(nb_neurons)]
    
    def add_weights(self):
        self.weights = [neuron.weights for neuron in self.neurons]
        self.weights = np.array(self.weights)

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
df = pd.read_csv('data_training.csv', names=columns)
# numerize column_1
df['column_1'] = df['column_1'].map({'M': 1, 'B': 0})
# split dataset
target = df.column_1
# create training and testing vars
X_train_0, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
X_train = X_train_0.drop('column_1', axis=1)
X_test.to_csv('test.csv', index=False)
X_train_scale = scale_data(X_train)
# Network
epochs = 1000    # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = X_train.shape[1] + 1, X_train.shape[1] + 1, 1
L = 0.03    # learning rate
network = Network()
network.add_layer(inputLayerSize, 'input')
network.add_layer(hiddenLayerSize, 'hidden_1')
network.add_layer(hiddenLayerSize, 'hidden_2')
network.add_layer(outputLayerSize, 'output')

# Initialize Input layer
X = np.reshape(X_train_scale.values, (X_train.shape[0], X_train.shape[1]))
add = np.ones((X_train.shape[0], 1))
X = np.concatenate((X, add), axis=1)
Y = np.reshape(y_train.values, (X_train.shape[0], 1))
for index, neuron in enumerate(network.layers[0].neurons):
    neuron.output = X[index]
network.layers[0].add_outputs()
network.layers[1].add_weights()
network.layers[2].add_weights()
network.layers[3].add_weights()
### outlayer ###
network.layers[3].weights = np.reshape(network.layers[3].weights, (X_train.shape[1] + 1, 1))

for i in range(epochs):
    # forward propagation
    ### Hidden layer 1 ###
    z_h_1 = np.dot(X, network.layers[1].weights)
    network.layers[1].outputs = sigmoid(z_h_1)
    ### Hidden layer 2 ###
    z_h_2 = np.dot(X, network.layers[2].weights)
    network.layers[2].outputs = sigmoid(z_h_2)
    ###output###
    z_o = np.dot(network.layers[2].outputs, network.layers[3].weights)
    network.layers[3].outputs = sigmoid(z_o)
    # Calculate the error
    E = Y - network.layers[3].outputs                                           # how much we missed (error)
    err = ((1 / 2) * (np.power((E), 2)))
    # Backpropagation
    ## Output layer
    delta_z_o = sigmoid_(network.layers[3].outputs)
    delta_w13 = network.layers[2].outputs
    delta_output_layer = np.dot(delta_w13.T,(E * delta_z_o))
    ## Hidden layer 2
    delta_a_h_2 = np.dot(E * delta_z_o, network.layers[3].weights.T)
    delta_z_h_2 = sigmoid_(network.layers[2].outputs)
    delta_w12 = network.layers[1].outputs
    delta_hidden_layer_2 = np.dot(delta_w12.T, delta_a_h_2 * delta_z_h_2)
    ## Hidden layer 1
    delta_a_h_1 = np.dot(E * delta_z_h_2, network.layers[2].weights.T)
    delta_z_h_1 = sigmoid_(network.layers[1].outputs)
    delta_w11 = X
    delta_hidden_layer_1 = np.dot(delta_w11.T, delta_a_h_1 * delta_z_h_1)

    network.layers[1].weights = network.layers[1].weights + L * delta_hidden_layer_1
    network.layers[2].weights = network.layers[2].weights + L * delta_hidden_layer_2
    network.layers[3].weights =  network.layers[3].weights + L * delta_output_layer
    print("epoch {}/{} - loss: {} - val_loss: {}".format(i + 1, epochs, err.sum(), 0))

Y_predict = []
# print(list(network.layers[3].outputs))
for x in network.layers[3].outputs:
    if x >= 0.5:
        Y_predict.append(1)
    else:
        Y_predict.append(0)
Y_predict = np.array(Y_predict)
weights_dic = {}
weights_dic['hidden_1'] = network.layers[1].weights
weights_dic['hidden_2'] = network.layers[2].weights
weights_dic['output'] = network.layers[3].weights
weights_dic['mean'] = X_train_0.mean().tolist()
weights_dic['std'] = X_train_0.std().tolist()
np.save('weights.npy', weights_dic)
