import numpy as np

import pickle
import gzip

def softmax(x, derivative):
    exps = np.exp(x)
    if derivative == True:
        return exps/np.sum(exps,axis=0) * (1-exps/np.sum(exps,axis=0))
    return exps/np.sum(exps,axis=0)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function (take sigmoid func as input)
def sigmoid_derivative(x):
    return x * (1 - x)

class feedForwardNN:
    def __init__(self, nbr_input, nbr_hidden, nbr_output):
        # Initialize the weights with random values
        self.weights_input_hidden = np.random.randn(nbr_input, nbr_hidden)/np.sqrt(nbr_hidden)
        self.weights_hidden_output = np.random.randn(nbr_hidden, nbr_output)/np.sqrt(nbr_output)

        self.b_hid = 0
        self.b_out = 0
    
    def feedForward(self, input):
        self.hidden_activations = sigmoid(np.dot(input, self.weights_input_hidden) + self.b_hid)
        self.output_activations = (np.dot(self.hidden_activations, self.weights_hidden_output) + self.b_out)
        
        return softmax(self.output_activations,False)
    
    def backward(self, inputs, targets, learning_rate):
         # Calculate the error at the output layer
        target = np.zeros(10)
        target[targets] = 1

        output_error = target - softmax(self.output_activations,False)
        # Calculate the gradient at the output layer using the error and the derivative of sigmoid
        output_delta = output_error * softmax(self.output_activations,True)

        # Calculate the error at the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        # Calculate the gradient at the hidden layer using the error and the derivative of sigmoid
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_activations)

        # Update the weights for both layers
        weights_output_derivative = np.outer(self.hidden_activations.T, output_delta)
        weights_input_derivative = np.outer(inputs.T,(hidden_delta))
        # Update bias terms
        b_out = np.sum(output_delta, axis=0)
        b_hid = np.sum(hidden_delta, axis=0)

        return [weights_output_derivative, weights_input_derivative, b_out, b_hid]

    
    def update_params(self, w_outputderivative, w_inputderivative, b_outputderivative, b_inputderivative, learning_rate, batch_size):
        self.weights_hidden_output = np.add(self.weights_hidden_output,(w_outputderivative / batch_size)*learning_rate)
        self.weights_input_hidden = np.add(self.weights_input_hidden, (w_inputderivative / batch_size)*learning_rate)
        self.b_out = np.add(self.b_out, (b_outputderivative / batch_size)*learning_rate)
        self.b_hid = np.add(self.b_hid, (b_inputderivative / batch_size)*learning_rate)
    
    def stochGrad(self, inputs, targets, epochs, batch_size):
        for epoch in range(1, epochs + 1):
            index_vector = np.random.permutation(len(targets))
            print(f"Starting epoch {epoch}...")
            
            for i in range(int(len(index_vector)/batch_size)):
                # Forward pass
                indices = index_vector[range(i*batch_size, (i+1)*batch_size)] # Ta ut en batch
                
                # from slides p 72, 1/(c * (1 + j)) where c > 0
                #learning_rate = 1/(2 * (i + 1))
                learning_rate = 1
                
                w_outputderivative = np.zeros(self.weights_hidden_output.shape)
                w_inputderivative = np.zeros(self.weights_input_hidden.shape)
                b_outputderivative = 0
                b_inputderivative = 0
                for k in indices:
                    self.feedForward(inputs[k]) # predicta bilder i batch
                    # Backpropagation
                    [dwout, dwhid, dbout, dbhid] = self.backward(inputs[k], targets[k], learning_rate)
                    w_outputderivative = np.add(w_outputderivative, dwout)
                    w_inputderivative = np.add(w_inputderivative, dwhid)
                    b_outputderivative = np.add(b_outputderivative, dbout)
                    b_inputderivative = np.add(b_inputderivative, dbhid)

                self.update_params(w_outputderivative, w_inputderivative, b_outputderivative, b_inputderivative, learning_rate, batch_size)

            self.test(test_inputs, test_targets, epoch)

    def test(self, test_inputs, test_targets, epoch = 1):
        stat = 0;
        for i in range(len(test_inputs)):
            prediction = self.feedForward(test_inputs[i])
            if np.argmax(prediction) == test_targets[i]: stat += 1
        
        print(f"Epoch: {epoch} \t{stat/len(test_inputs) * 100}%, total found: {stat}")

def main():
    global test_inputs
    global test_targets

    network = feedForwardNN(784, 30, 10)

    with gzip.open("mnist.pkl.gz", 'rb') as f:
        mnist = pickle.load(f, encoding="latin1")

    train_inputs = mnist[0][0]
    train_targets = mnist[0][1]

    test_inputs = mnist[2][0]
    test_targets = mnist[2][1]

    network.stochGrad(train_inputs,train_targets,4,10)

if __name__ == "__main__":
    main()