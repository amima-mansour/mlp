from random import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from  matplotlib import pyplot as plt
from tools import predict, final_outputs, sigmoid, softmax
import argparse

def standardize(vector, mean, std):
    return (vector - mean) / std

def sigmoid_(x): return (x * (1 - x))

def mean_square_error(Y, Y_predict):
    loss = np.sum((Y - Y_predict) ** 2)
    return loss / Y_predict.shape[0]


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
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Set Dataset name to train")
args = parser.parse_args()
#dataset columns
columns = []
for i in range(32):
    columns.append('column_' + str(i))
try:
    df = pd.read_csv(args.dataset, names=columns)
    # numerize column_1
    df['M'] = df['column_1'].map({'M': 1, 'B': 0})
    df['B'] = df['column_1'].map({'M': 0, 'B': 1})
    # split dataset
    target = df['M']
    X_train_0, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=1)
    X_train_0, X_val, y_train, y_val = train_test_split(X_train_0, y_train, test_size=0.2, random_state=1)
    ## test data
    X_test = X_test.drop('B', axis=1)
    X_test = X_test.drop('M', axis=1)
    X_test.to_csv('test.csv', index=False)
    ## train data
    X_train = X_train_0.drop(['column_1','M'], axis=1)
    Y_M = np.reshape(y_train.values, (X_train.shape[0], 1))
    Y_B = np.reshape(X_train.B.values, (X_train.shape[0], 1))
    X_train = X_train.drop('B', axis=1)
    mean, std = X_train.mean().tolist(), X_train.std().tolist()
    X_train_scale = scale_data(X_train)
    X_train = np.reshape(X_train_scale.values, (X_train.shape[0], X_train.shape[1]))
    Y_train = np.concatenate((Y_M, Y_B), axis=1)
    #### Validation data
    X_val = X_val.drop(['column_1', 'M'], axis=1)
    Y_M = np.reshape(y_val.values, (X_val.shape[0], 1))
    Y_B = np.reshape(X_val.B.values, (X_val.shape[0], 1))
    X_val = X_val.drop('B', axis=1)
    X_val_scale = standardize(X_val, mean, std)
    X_val = np.reshape(X_val_scale.values, (X_val.shape[0], X_val.shape[1]))
    Y_val = np.concatenate((Y_M, Y_B), axis=1)
    #### Shuffle
    X, Y = shuffle(X_train, Y_train, random_state=np.random.RandomState())
    X_val, Y_val = shuffle(X_val, Y_val, random_state=np.random.RandomState())
    # Network
    epochs = 250    # Number of iterations
    inputLayerSize, hiddenLayerSize_1, hiddenLayerSize_2, outputLayerSize = X_train.shape[1], 16, 8,2
    L = 0.3  # learning rate
    network = Network()
    network.add_layer(inputLayerSize, 'input')
    network.add_layer(hiddenLayerSize_1, 'hidden_1')
    network.add_layer(hiddenLayerSize_2, 'hidden_2')
    network.add_layer(outputLayerSize, 'output')
    for index, neuron in enumerate(network.layers[0].neurons):
        neuron.output = X[index]
    network.layers[0].add_outputs()
    network.layers[1].add_weights()
    network.layers[2].add_weights()
    network.layers[3].add_weights()
    network.layers[1].add_bias()
    network.layers[2].add_bias()
    network.layers[3].add_bias()
    val_loss, loss = [], []
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
        d = delta_a_h_1 * delta_z_h_1
        delta_w11 = X
        dw_h_1 = np.dot(d.T, delta_w11) / X.shape[0]
        db_h_1 = np.sum(d, axis=0, keepdims=True) / X.shape[0]
        ### Update weights and bias
        network.layers[1].weights -= L * dw_h_1
        network.layers[2].weights -= L * dw_h_2
        network.layers[3].weights -= L * dw_o
        network.layers[1].bias -= L * db_h_1
        network.layers[2].bias -= L * db_h_2
        network.layers[3].bias -= L * db_o
        #### Loss function
        Y_predict = final_outputs(network.layers[3].outputs)
        loss.append(mean_square_error(Y[:, 0], Y_predict))
        Y_val_predict = predict(X_val, network.layers[1].weights, network.layers[1].bias, network.layers[2].weights,
            network.layers[2].bias, network.layers[3].weights, network.layers[3].bias)
        val_loss.append(mean_square_error(Y_val[:, 0], Y_val_predict))
        print("epoch {}/{} - loss: {:.10f} - val_loss: {:.10f}".format(i + 1, epochs, loss[i], val_loss[i]))
    plt.plot(range(1, epochs + 1), loss, 'g--', label='loss')
    plt.plot(range(1, epochs + 1), val_loss, 'r--', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.legend()
    plt.show()
    print('Accuracy training = {:.2f}'.format(accuracy_score(Y[:, 0], Y_predict)))
    print('Accuracy validation = {:.2f}'.format(accuracy_score(Y_val[:, 0], Y_val_predict)))
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
except:
    print("Error dataset")
