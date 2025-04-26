import numpy as np

chars = ['d', 'o', 'g', 's']
char_to_int = {ch: idx for idx, ch in enumerate(chars)}
int_to_char = {idx: ch for idx, ch in enumerate(chars)}

def generate_data():
    X, y = [], []
    for i in range(len(chars) - 1):
        X.append([char_to_int[chars[i]]])
        y.append([char_to_int[chars[i+1]]])
    return np.array(X), np.array(y)

X, y = generate_data()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def feedforward(seq_x, layers):
    i_weight, h_weight, h_bias, o_weight, o_bias = layers[0]
    hidden = sigmoid(np.dot(seq_x, i_weight) + h_bias)
    output = sigmoid(np.dot(hidden, o_weight) + o_bias)
    return hidden, output

def backpropagation(layers, X, y, lr, hidden):
    i_weight, h_weight, h_bias, o_weight, o_bias = layers[0]
    o_weight_grad = np.dot(hidden.T, (y - hidden))
    o_bias_grad = np.sum(y - hidden)
    h_grad = np.dot((y - hidden), o_weight.T) * sigmoid_derivative(hidden)
    i_weight_grad = np.dot(X.T, h_grad)
    h_weight_grad = np.dot(hidden.T, h_grad)
    h_bias_grad = np.sum(h_grad, axis=0)

    i_weight -= lr * i_weight_grad
    h_weight -= lr * h_weight_grad
    h_bias -= lr * h_bias_grad
    o_weight -= lr * o_weight_grad
    o_bias -= lr * o_bias_grad

    return [[i_weight, h_weight, h_bias, o_weight, o_bias]]

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def init_params():
    layers = []
    np.random.seed(0)
    k = 1 / np.sqrt(4)
    i_weight = np.random.rand(1, 4) * 2 * k - k
    h_weight = np.random.rand(4, 4) * 2 * k - k
    h_bias = np.random.rand(1, 4) * 2 * k - k
    o_weight = np.random.rand(4, len(chars)) * 2 * k - k
    o_bias = np.random.rand(1, len(chars)) * 2 * k - k
    layers.append([i_weight, h_weight, h_bias, o_weight, o_bias])
    return layers

epochs = 1000
lr = 1e-4

layers = init_params()

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(X.shape[0]):
        seq_x = X[i:(i+1), :]
        seq_y = y[i:(i+1), :]
        
        hidden, output = feedforward(seq_x, layers)
        epoch_loss += mse_loss(seq_y, output)
        
        layers = backpropagation(layers, seq_x, seq_y, lr, hidden)

    if epoch % 50 == 0:
        print(f"Epoch: {epoch} Loss: {epoch_loss / len(X)}")

def predict_next_char(char):
    int_char = char_to_int[char]
    input_data = np.array([[int_char]])
    hidden, prediction = feedforward(input_data, layers)
    predicted_char = int_to_char[np.argmax(prediction)]
    return predicted_char

test_char = 'd'
print(f"Next character after '{test_char}' is '{predict_next_char(test_char)}'")
