import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# define training data
# rule: return 1 if first input is 1
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

# assign weights

np.random.seed(1)

weights = 2 * np.random.random((3, 1)) - 1

print("Weights before training: ", weights)

for i in range(100000):
    input_layer = training_inputs

    # calculate an output
    outputs = sigmoid(np.dot(input_layer, weights))

    # find how wrong the calculated output was against the expected output
    error = training_outputs - outputs

    # calculate adjustment
    adjustment = error * sigmoid_derivative(outputs)

    weights += np.dot(input_layer.T, adjustment)

print("Weights after training: ", weights)
print("outputs after training: ", outputs)
