import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights and bias
np.random.seed(42)
weights_input_to_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_to_output = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, output_size)

# Training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])  # output for XOR

# Training parameters
learning_rate = 0.5
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_to_hidden) + bias_hidden

    hidden_layer_output = sigmoid(hidden_layer_input)

    ouptut_layer_input = (
        np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    )

    predicted_output = sigmoid(ouptut_layer_input)

    # Loss calculation
    error = Y - predicted_output
    loss = np.mean(np.square(error))

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_to_output.T)

    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating weights and biases
    weights_hidden_to_output += (
        hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    )
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_input_to_hidden += X.T.dot(d_hidden_layer) * learning_rate

    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")


# Testing
print("Predictions:")
print(predicted_output)
