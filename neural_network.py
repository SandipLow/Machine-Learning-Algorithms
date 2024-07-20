import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# activation functions
def tanh(x) :
    return np.tanh(x)

def relu(x) :
    return np.maximum(x, 0)

def softmax(x) :
    expX = np.exp(x)
    return expX/np.sum(expX)


# Derivatives
def derivative_tanh(x) :
    return 1 -  np.power(np.tanh(x), 2)

def derivative_relu(x) :
    return np.array(x > 0, dtype=np.float32)



class Layer :
    def __init__(self, neurons: int, prev) -> None :
        self.neuronCount = neurons
        self.prev = prev
        self.bias = np.random.randn(1, neurons)
        self.value = np.zeros((1, neurons))

    def setValue(self, weights: np.ndarray) -> None :
        self.value = softmax( np.dot(self.prev.value, weights) + self.bias )


class InpLayer :
    def __init__(self, neurons: int) -> None :
        self.neuronCount = neurons
        self.value = np.zeros((1, neurons))

    def setValue(self, X: np.ndarray) -> None :
        self.value = X


class Network :
    def __init__(self, inputs: int, outputs: int, hidden: int) -> None:
        # Layers
        self.input   = InpLayer(inputs)
        self.hidden1 = Layer(hidden, self.input)
        self.hidden2 = Layer(hidden, self.hidden1)
        self.hidden3 = Layer(hidden, self.hidden2)
        self.output  = Layer(hidden, self.hidden3)
        # Weights
        self.Wi1 = np.random.randn(inputs, hidden)
        self.W12 = np.random.randn(hidden, hidden)
        self.W23 = np.random.randn(hidden, hidden)
        self.W3o = np.random.randn(hidden, outputs)


    def calculate_cost(self, Y: np.ndarray, Y_predicted: np.ndarray) -> int :
        (n, m) = Y.shape
        cost = -(1/m)*np.sum(Y*np.log(Y_predicted))
        return cost


    def forwardPropagation(self, X) :
        self.input.setValue(X)
        self.hidden1.setValue(self.Wi1)
        self.hidden2.setValue(self.W12)
        self.hidden3.setValue(self.W23)
        self.output.setValue(self.W3o)


    def backPropagation(self, X, Y) :
        (n, m) = X.shape

        dO = Y-self.output.value
        dW3o = (1/m)*np.dot(dO, self.hidden1.value.T)
        dBo = (1/m)*np.sum(dO, axis=1, keepdims=True)

        dH3 = (1/m)*np.dot(self.W3o.T, dO)*derivative_relu(self.hidden3.value)
        dW23 = (1/m)*np.dot(dH3, self.hidden2.value.T)
        dB3 = (1/m)*np.sum(dH3, axis=1, keepdims=True)

        dH2 = (1/m)*np.dot(self.W23.T, dH3)*derivative_relu(self.hidden2.value)
        dW12 = (1/m)*np.dot(dH2, self.hidden1.value.T)
        dB2 = (1/m)*np.sum(dH2, axis=1, keepdims=True)

        dH1 = (1/m)*np.dot(self.W12.T, dH2)*derivative_relu(self.hidden1.value)
        dWi1 = (1/m)*np.dot(dH1, self.input.value.T)
        dB1 = (1/m)*np.sum(dH1, axis=1, keepdims=True)

        return {
            "dW3o": dW3o,
            "dW23": dW23,
            "dW12": dW12,
            "dWi1": dWi1,
            "dBo":  dBo,
            "dB3":  dB3,
            "dB2":  dB2,
            "dB1":  dB1,
        }
    
    def updateParameters(self, gradients, learning_rate) :
        dW3o = gradients["dW3o"]
        dW23 = gradients["dW23"]
        dW12 = gradients["dW12"]
        dWi1 = gradients["dWi1"]
        dBo  = gradients["dBo"]
        dB3  = gradients["dB3"]
        dB2  = gradients["dB2"]
        dB1  = gradients["dB1"]

        self.Wi1          -= learning_rate*dWi1
        self.W12          -= learning_rate*dW12
        self.W23          -= learning_rate*dW23
        self.W3o          -= learning_rate*dW3o
        self.hidden1.bias -= learning_rate*dB1
        self.hidden2.bias -= learning_rate*dB2
        self.hidden3.bias -= learning_rate*dB3
        self.output.bias  -= learning_rate*dBo


    def train(self, X, Y, learning_rate, iterations) :
        cost_list = []

        for i in range(iterations) :
            self.forwardPropagation(X)
            cost = self.calculate_cost(Y, self.output.value)
            gradients = self.backPropagation(X, Y)
            self.updateParameters(gradients, learning_rate)
            cost_list.append(cost)

            if (i%10 == 0) :
                print(f"Cost after  {i} iteratiion : {cost}")

        t = np.arange(0, iterations)
        plt.plot(t, cost_list)
        plt.show()





# Testing
X_train = np.loadtxt('data/train_X.csv', delimiter = ',').T
Y_train = np.loadtxt('data/train_label.csv', delimiter = ',').T

X_test = np.loadtxt('data/test_X.csv', delimiter = ',').T
Y_test = np.loadtxt('data/test_label.csv', delimiter = ',').T

(i, m) = X_train.shape
(o, m) = Y_train.shape


model = Network(i, o, 20)
model.train(X_train.T, Y_train.T, 0.01, 1000)






