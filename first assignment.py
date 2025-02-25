import random 

def tanh(x):
    return (2.71828 ** x - 2.71828 ** -x) / (2.71828 ** x + 2.71828 ** -x)

input_neurons = 4
hidden_neurons = 3
output_neurons = 2

weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(input_neurons)] for _ in range(hidden_neurons)]
weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(hidden_neurons)] for _ in range(output_neurons)]

inputs = [random.uniform(-1, 1) for _ in range(input_neurons)]

b1 = 0.5
b2 = 0.7

hidden_layer = [tanh(sum(w * x for w, x in zip(weights, inputs)) + b1) for weights in weights_input_hidden]

outputs = [tanh(sum(w * h for w, h in zip(weights, hidden_layer)) + b2) for weights in weights_hidden_output]


print("Output of the network:", outputs)
