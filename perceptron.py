import numpy as np


class PerceptronNetwork:

    def __init__(self):
        np.random.seed(1)

        # create weights psuedo-randomly between -1 and 1 with a mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # function to calculate the value of the sigmoid function at a given point x
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # function to calculate the slope of the sigmoid function at a given point x
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # function to train the neural network
    def train(self, training_inputs, training_outputs, training_it):
        for it in range(training_it):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # make the adjustments (back propagation)
            self.synaptic_weights += adjustments

    def think(self, inputs):
        # need to make sure inputs are floats
        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":
    perceptron_network = PerceptronNetwork()
    print("Random Synaptic Weights:\n", perceptron_network.synaptic_weights)

    # our training data
    training_inputs = np.array([[0, 1, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    # run the neuron with the training data 10000 times
    perceptron_network.train(training_inputs, training_outputs, 100000)

    # results printed to console

    print("Synaptic Weights Post Training:\n", perceptron_network.synaptic_weights)

    # Allow the user to test the network
    input_1 = str(input("Input 1: "))
    input_2 = str(input("Input 2: "))
    input_3 = str(input("Input 3: "))

    print("Testing neuron with: ", input_1, input_2, input_3)
    print("Neuron output: ", perceptron_network.think(np.array([input_1, input_2, input_3])))


