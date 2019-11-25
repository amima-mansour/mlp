from random import random 
import numpy as np

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid

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


epochs = 20000    # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 2, 1
L = .1    # learning rate

network = Network()
network.add_layer(inputLayerSize, 'input')
network.add_layer(hiddenLayerSize, 'hidden')
network.add_layer(outputLayerSize, 'output')

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])



# network.layers[0].neurons = X

i = 0
x = X[1]
for index, neuron in enumerate(network.layers[0].neurons):
    neuron.output = x[index]
network.layers[0].add_outputs()
network.layers[1].add_weights()
network.layers[2].add_weights()
# A checker PRQ
network.layers[2].weights = np.reshape(network.layers[2].weights, (2,1))

for i in range(epochs):
    network.layers[1].outputs = sigmoid(np.dot(X, network.layers[1].weights)) # hidden layer results
    network.layers[2].outputs = np.dot(network.layers[1].outputs, network.layers[2].weights)   # output layer, no activation
    E = Y - network.layers[2].outputs                                                                   # how much we missed (error)
    dZ = E * L                                                                                 # delta Z
    network.layers[2].weights +=  network.layers[1].outputs.T.dot(dZ)                          # update output layer weights
    dH = dZ.dot(network.layers[2].weights.T) * sigmoid_(network.layers[1].outputs)             # delta H
    network.layers[1].weights += X.T.dot(dH)                          # update hidden layer weights

print(network.layers[2].outputs)

### 2pbs to check ####
### with reshape ####
### with input layer => how to use it ###