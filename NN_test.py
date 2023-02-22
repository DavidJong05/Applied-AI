import numpy as np
import random

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.delta = 0
        self.output = 0
        self.sum = 0

    def __str__(self):
        return f"Neuron Weights: {[round(weight, 2) for weight in self.weights]} Bias: {round(self.bias, 2)}"

def sig(x):
    return 1/(1 + np.exp(-x))

def sig_grad(x):
    e_x = np.exp(-x)
    return e_x/(e_x+1)**2

class NeuralNetwork:
    def __init__(self, amount_of_layers: int, layer_depths: list) -> None:
        '''
        :param amount_of_layers: amount of layers in the matrix/neural network
        :param layer_depths: The depht of a layer, a list with the depth of each layer defined as integers
        Creates a matrix with neurons in each layer with random weights and bias.
        First layer is input layer
        '''
        if amount_of_layers != len(layer_depths):
            raise SyntaxError(f"Layers not equal to length of layer depth list")

        self.matrix = [[] for n in range(0, amount_of_layers)] # Initialize empty matrix

        #fill matrix with random weights from -1 to 1
        for i in range(0, amount_of_layers):
            for j in range(0, layer_depths[i]):
                if i == 0: # first layer is input layer so skip
                    self.matrix[i].append(Neuron([0 for j in range(layer_depths[i])], 0))
                else:
                    self.matrix[i].append(Neuron([random.uniform(-1, 1) for j in range(layer_depths[i-1])], random.uniform(-1, 1))) # rest of layers with random values (-1,1)

    def __str__(self):
        return '\n''\n'.join(['\n'.join([str(neuron) for neuron in layer]) for layer in self.matrix])

    def feed_forward(self):
        '''
        Assigns a designated output to each neuron based on formula:
        All weigths multiplied by all outputs of the previous layer, next to this add the bias.
        the result needs to put in the sigmoid function to obtain new output
        '''
        for index, layer in enumerate(self.matrix):
            if index == 0: # skip the input layer
                continue
            for neuron in layer:
                result = 0
                for depth, weight in enumerate(neuron.weights):
                    result += weight * self.matrix[index - 1][depth].output # all(Wi * aiL-1) <-  All weights of neuron * all outputs of prev layer
                neuron.sum = neuron.bias + result # Zi = bi + all(Wi * aiL-1) <- All weights * all outputs + bias of a layer
                neuron.output = sig(neuron.sum) # ai = sig(Zi)


    def back_prop(self, expected: list):
        '''
        @:param expected: expected outputs list
        Calculates the delta between the actual result compared to the predicted outcome, also known as the mistake rate.
        The first delta(outputs of last layer) is done differently because the expected result gets compared with the actual result.
        the delta is calculated by putting the weights + previous outputs(neuron.sum) in the sigmoid gradient function,
        multiplied by the mistake rate(expected - output). Do this for all previous layers
        '''
        # Backpropagate the last output layer
        for neuron, expected_result in zip(self.matrix[-1], expected):
            neuron.delta = sig_grad(neuron.sum) * (expected_result - neuron.output) # delta_j(last) = Sig_grad(Zj) * (yj-aj)

        # Backpropagate the previous layers
        for index, layer in enumerate(reversed((self.matrix[1:-1]))):
            for depth, neuron in enumerate(layer):
                result = 0
                for prev_neuron in self.matrix[index - 1]:
                    result += prev_neuron.delta * prev_neuron.weights[depth]
                neuron.delta = sig_grad(neuron.sum) * result # delta_i = sig_grad(ZiL-1) * (delta_j * Wij)


    def update_weights(self, learning_rate: int):
        '''
        @:param learning_rate: The learning rate is a parameter that controls how much to change the network in response to the estimated error each time the weights are updated.
        After have done back propagation, you can update the weights by knowing the mistake rate
        starting from the end
        The weights are updated by multiplying the learning rate by the delta by the previous layer output
        The bias is updated by the learning rate multiplied by the delta
        '''
        for index, layer in enumerate(reversed(self.matrix[1:])):
            for neuron in layer:
                for depth, weight in enumerate(neuron.weights):
                    reverse = len(self.matrix) - 2 # looping reversed through matrix[1:] to get previous layer index
                    neuron.weights[depth] += learning_rate * neuron.delta * self.matrix[reverse - index][depth].output # Wij += lr * delta_j * aiL-1
                    neuron.bias += learning_rate * neuron.delta # bj += lr * delta_j


    def train(self, inputs: list, outputs: list, epochs: int = 1, lr: int = 0.01):
        '''
        :param inputs: list of the inputs going into the network
        :param outputs: outputs are the expected outputs
        :param epochs: one cycle of training the network, atleast once
        :param lr: learning rate, described in the update_weights function
        '''
        for epoch in range(0, epochs):
            for index, data_point in enumerate(inputs):
                for input, neuron in zip(data_point, self.matrix[0]): # put inputs in first layer
                    neuron.output = float(input)

                self.feed_forward()
                self.back_prop(outputs[index])
                self.update_weights(lr)

                #testing purposes
                #network_outputs = [neuron.output for neuron in self.matrix[-1]] # outputs of the last layer
                #print("inputs: ",inputs[index])
                #print("e_outputs: ",outputs[index])
                #print("outputs: ",network_outputs)
        print("Training finished")

    def test(self, inputs: list, expected_outputs: list):
        '''
        @:param inputs: list of the input data going into the network
        @:param expected_outputs: list of the expected outputs
        Tests the NN after its trained. by doing 1 feed_forward with the given inputs.
        compares the expected_outputs with the actual network_outputs
        '''
        counter = 0
        for index, input in enumerate(inputs):
            for input, neuron in zip(input, self.matrix[0]):
                neuron.output = float(input)

            self.feed_forward()

            network_outputs = [neuron.output for neuron in self.matrix[-1]]
            rounded_outputs = [round(output) for output in network_outputs]
            if expected_outputs[index] == rounded_outputs:
                counter += 1

        print(f"Accuracy: {round((counter/len(expected_outputs))*100)}")


xor_network = NeuralNetwork(3, [2, 2, 1])
inputs = [[0, 0],
          [1, 0],
          [0, 1],
          [1, 1]]
outputs = [[0],
           [1],
           [1],
           [0]]

#print("XOR Network:")
#xor_network.train(inputs, outputs, 9000, 0.05)
#print("Testing:")
#xor_network.test(inputs, outputs)

label_list = []
data = np.genfromtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
labels = np.genfromtxt('iris.data', delimiter=',', names=True, dtype=None, encoding=None, usecols=[4])
for label in labels:
    for l in label:
        if l == 'Iris-setosa':
            label_list.append([1,0,0])
        elif l == 'Iris-versicolor':
            label_list.append([0,1,0])
        elif l == 'Iris-virginica':
            label_list.append([0,0,1])

print("\nIris Network:")
iris_network = NeuralNetwork(3, [4, 4, 3])
iris_network.train(data[:130], label_list[:130], 3000, 0.01)
print("Testing:")
iris_network.test(data[:130], label_list[:130])