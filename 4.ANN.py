import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training function for the neural network
def train(X, y, epochs, learning_rate):
    input_layer_neurons = X.shape[1]
    hidden_layer_neurons = 2  # Number of hidden layer neurons
    output_neurons = 1  # Number of output neurons

    # Weight and bias initialization
    hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
    output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
    output_bias = np.random.uniform(size=(1, output_neurons))

    # Training the neural network
    for _ in range(epochs):
        # Forward propagation
        hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
        hidden_layer_activation = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_activation, output_weights) + output_bias
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

        # Updating weights and biases
        output_weights += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return hidden_weights, hidden_bias, output_weights, output_bias

# Prediction function
def predict(X, hidden_weights, hidden_bias, output_weights, output_bias):
    hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_activation = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activation, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    return predicted_output

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Parameters
epochs = 10000
learning_rate = 0.1

# Training the neural network
hidden_weights, hidden_bias, output_weights, output_bias = train(X, y, epochs, learning_rate)

# Testing the neural network
predictions = predict(X, hidden_weights, hidden_bias, output_weights, output_bias)
print("Predicted outputs:")
print(predictions)
