import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layer):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer = hidden_layer
        self.W1, self.W2, self.b1, self.b2 = self.init_params()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss(self, y_pred, Y):
        M = Y.shape[1]
        loss = - ((np.sum(Y * np.log(y_pred)) + np.sum((1 - Y) * np.log(1 - y_pred))) / M)
        return loss

    def init_params(self):
        W1 = np.random.randn(self.hidden_layer, self.input_size) * 0.01
        W2 = np.random.randn(self.output_size, self.hidden_layer) * 0.01
        b1 = np.random.randn(self.hidden_layer, 1)
        b2 = np.random.randn(self.output_size, 1)
        return W1, W2, b1, b2

    def forward_pass(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1

    def backward_pass(self, X, Y, A2, Z2, A1, Z1):
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.W2.T, dZ2) * self.d_sigmoid(Z1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        return dW1, dW2, db1, db2

    def accuracy(self, y_pred, y):
        m = y.shape[1]
        accuracy_n = np.sum(y_pred == y) / m
        return accuracy_n

    def predict(self, X):
        A2, _, _, _ = self.forward_pass(X)
        predictions = (A2 >= 0.5).astype(int)
        return predictions

    def update(self, dW1, dW2, db1, db2, alpha):
        self.W1 = self.W1-alpha * dW1
        self.W2 = self.W2-alpha * dW2
        self.b1 = self.b1-alpha * db1
        self.b2 = alpha * db2










var = 0.2
n = 800
class_0_a = var * np.random.randn(n // 4, 2)
class_0_b = var * np.random.randn(n // 4, 2) + (2, 2)
class_1_a = var * np.random.randn(n // 4, 2) + (0, 2)
class_1_b = var * np.random.randn(n // 4, 2) + (2, 0)
X = np.concatenate([class_0_a, class_0_b, class_1_a, class_1_b], axis=0)
Y = np.concatenate([np.zeros((n // 2, 1)), np.ones((n // 2, 1))])
X = X.T
Y = Y.T

# Train test split
ratio = 0.8
X_train = X[:, :int(n * ratio)]
Y_train = Y[:, :int(n * ratio)]

X_test = X[:, int(n * ratio):]
Y_test = Y[:, int(n * ratio):]

# Initialize neural network
input_size = 2
output_size = 1
hidden_layer = 10
nn = NeuralNetwork(input_size, output_size, hidden_layer)
