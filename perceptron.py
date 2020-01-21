import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# define training data
training_inputs = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 0, 1],
                            [1, 1, 0],
                            [0, 1, 1],
                            [1, 1, 1]])

training_outputs = np.array([[0, 0, 0, 0, 1, 0, 0, 1]]).T

# assign weights

np.random.seed(1)

weights = 2 * np.random.random((3, 1)) - 1

print("Weights: ", weights)

for i in range(1):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, weights))

print("outputs after training: ", outputs)
