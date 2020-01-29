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
