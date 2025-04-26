def tanh(x):
    return (2.71828 ** x - 2.71828 ** -x) / (2.71828 ** x + 2.71828 ** -x)

def tanh_derivative(x):
    return 1 - x ** 2

input_neurons = 2
hidden_neurons = 2
output_neurons = 2
learning_rate = 0.5

weights_input_hidden = [[0.15, 0.25], [0.20, 0.30]]
weights_hidden_output = [[0.40, 0.50], [0.45, 0.55]]

inputs = [0.05, 0.10]
targets = [0.01, 0.99]

b1 = 0.35
b2 = 0.60

hidden_layer = []
for i in range(hidden_neurons):
    net_h = 0
    for j in range(input_neurons):
        net_h += weights_input_hidden[i][j] * inputs[j]
    net_h += b1
    hidden_layer.append(tanh(net_h))

outputs = []
for i in range(output_neurons):
    net_o = 0
    for j in range(hidden_neurons):
        net_o += weights_hidden_output[i][j] * hidden_layer[j]
    net_o += b2
    outputs.append(tanh(net_o))

errors = []
for i in range(output_neurons):
    errors.append(targets[i] - outputs[i])

output_gradients = []
for i in range(output_neurons):
    output_gradients.append(errors[i] * tanh_derivative(outputs[i]))

hidden_errors = [0] * hidden_neurons
for i in range(hidden_neurons):
    for j in range(output_neurons):
        hidden_errors[i] += weights_hidden_output[j][i] * output_gradients[j]

hidden_gradients = []
for i in range(hidden_neurons):
    hidden_gradients.append(hidden_errors[i] * tanh_derivative(hidden_layer[i]))

for i in range(output_neurons):
    for j in range(hidden_neurons):
        weights_hidden_output[i][j] += learning_rate * output_gradients[i] * hidden_layer[j]

for i in range(hidden_neurons):
    for j in range(input_neurons):
        weights_input_hidden[i][j] += learning_rate * hidden_gradients[i] * inputs[j]

print("Updated Weights:")
print(f"w1: {weights_input_hidden[0][0]}, w2: {weights_input_hidden[1][0]}, w3: {weights_input_hidden[0][1]}, w4: {weights_input_hidden[1][1]}")

print("Updated Weights:")
print(f"w5: {weights_hidden_output[0][0]}, w6: {weights_hidden_output[1][0]}, w7: {weights_hidden_output[0][1]}, w8: {weights_hidden_output[1][1]}")

print("Output of the network after backpropagation:", outputs)
