# Logistic Regression :
# Coding Lane Logistic Regression Playlist : https://www.youtube.com/playlist?list=PLuhqtP7jdD8Chy7QIo5U0zzKP8-emLdny
# Implementaion PyNB : https://github.com/Jaimin09/Coding-Lane-Assets/blob/main/Logistic%20Regression%20in%20Python%20from%20Scratch/Logistic%20Regression%20-%20Titanic.ipynb
# Code With Harry : https://www.youtube.com/watch?v=eL6dukE4f-4&t=1746s

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

class LogisticRegression :
    def __init__(self) :
        self.weights = None
        self.bias = None

    def _sigmoid(self, x) :
        return 1/(1 + np.exp(-x))

    def fit(self, X, Y, learning_rate=0.001, iterations=10000) :
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0
        Y = Y.reshape(m, 1)

        costlist = []

        for i in range(iterations) :
            _Y = np.dot(X, self.weights) + self.bias
            Y_Predicted = self._sigmoid(_Y)

            cost = -(1/m)*np.sum( Y*np.log(Y_Predicted) + (1-Y)*np.log(1-Y_Predicted) )

            dW = (1/m)*np.dot(X.T, Y_Predicted-Y)
            dB = (1/m)*np.sum(Y_Predicted-Y)

            self.weights = self.weights - learning_rate*dW
            self.bias = self.bias - learning_rate*dB

            costlist.append(cost)

            if(i%(iterations/10) == 0):
                print("cost after ", i, "iteration is : ", cost)

        plt.plot(np.arange(iterations), costlist)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

    def accuracy(self, X, Y) :
        _Y = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(_Y)
        
        A = A > 0.5
        
        A = np.array(A, dtype = 'int64')
        
        acc = (1 - np.sum(np.absolute(A - Y))/Y.shape[1])*100
        
        print("Accuracy of the model is : ", round(acc, 2), "%")

    def predict(self, X) :
        _Y = np.dot(X, self.weights) + self.bias
        Y_Predicted = self._sigmoid(_Y)

        return Y_Predicted > 0.5


iris = datasets.load_iris()
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int64)

skModel = linear_model.LogisticRegression()
skModel.fit(X, y)

myModel = LogisticRegression()
myModel.fit(X, y, 0.05, 10000)

example = skModel.predict(([[2.6]]))
print(example)

example = myModel.predict(([[2.6]]))
print(example)