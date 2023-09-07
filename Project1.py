import numpy as np
from numpy import *; from matplotlib.pyplot import *

import pickle
import gzip

## hello there fuckers

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

class feedForwardNN:
    def __init__(self, nbr_input, nbr_hidden, nbr_output):

        # Initialize the weights with random values
        self.weights_input_hidden = np.random.rand(nbr_input, nbr_hidden);
        self.weights_hidden_output = np.random.rand(nbr_hidden, nbr_output);
        
        self.b_hid = np.zeros(nbr_hidden);
        self.b_out = np.zeros(nbr_output);
    
    def feedforward(self, input):
        hidden_activations = sigmoid(np.dot(input, self.weights_input_hidden) + self.b_hid)
        output_activations = sigmoid(np.dot(hidden_activations, self.weights_hidden_output) + self.b_out)
        
        return output_activations
    
    def backward(self, inputs, targets, learning_rate):
         # Calculate the error at the output layer
        output_error = targets - self.output_layer_output
        # Calculate the gradient at the output layer using the error and the derivative of sigmoid
        output_delta = output_error * sigmoid_derivative(self.output_layer_output)

        # Calculate the error at the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        # Calculate the gradient at the hidden layer using the error and the derivative of sigmoid
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)

        # Update the weights for both layers
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

    
    def stochDesc(self, inputs, targets, learning_rate, epochs, batch_size):
        for _ in range(epochs):
            index_vector = np.random.permutations(len(targets));
            for i in range(len(index_vector)/batch_size):
                # Forward pass
                indices = index_vector[range(i*batch_size, (i+1)*batch_size)]; # Ta ut en batch
                batchImages = [inputs[i] for i in indices]
                batchTargets = [targets[i] for i in indices]
                
                predictions= []
                for image in batchImages:
                    predictions.append(self.feedForward(image)) # predicta bilder i batch
                
                    
                # Backpropagation
                self.backward(predictions, targets[indices], learning_rate)



    
network = feedForwardNN(784, 30, 10)


with gzip.open("mnist.pkl.gz", 'rb') as f:
    mnist = pickle.load(f, encoding="latin1")

