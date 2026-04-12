import numpy as np

x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

learning_rate = 0.1
epochs = 10000

weights_input_hidden = np.random.rand(2,2)
bias_input_hidden = np.zeros((1,2))

weights_output_hidden = np.random.rand(2,1)
bias_output_hidden = np.zeros((1,1))

def calcSigmoid(x):
    return ( 1 / (1 + np.exp(-x)) )

for epoch in range(epochs):
    hidden_input = np.dot(x, weights_input_hidden) + bias_input_hidden
    hidden_output = calcSigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_output_hidden) + bias_output_hidden
    final_output = calcSigmoid(final_input)
    err = y - final_output
    d_output = err * (final_output * (1 - final_output))
    d_hidden = d_output.dot(weights_output_hidden.T) * (hidden_output * (1 - hidden_output))
    weights_output_hidden += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += x.T.dot(d_hidden) * learning_rate
    bias_output_hidden += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_input_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}\nError: {err}\nFinal output: {final_output}")

hidden_input = np.dot(x, weights_input_hidden) + bias_input_hidden
hidden_output = calcSigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_output_hidden) + bias_output_hidden
final_output = calcSigmoid(final_input)
print(final_output)