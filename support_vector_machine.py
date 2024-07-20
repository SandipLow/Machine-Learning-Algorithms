import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class SVM :
    def __init__(self) :
        self.weights = None
        self.bias = None

    def fit(self, X, Y, learning_rate, lambda_parameter, no_of_iterations) :
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        y_label = np.where(Y <= 0, -1, 1)

        costlist = []

        for i in range(no_of_iterations) :

            for row, x_i in enumerate(X) :
                condition = y_label[row]*(np.dot(x_i, self.weights) - self.bias) >= 1

                if condition :
                    self.weights -= learning_rate*(2*lambda_parameter*self.weights)
                else :
                    self.weights -= learning_rate*(2*lambda_parameter*self.weights - np.dot(x_i, y_label[row]))
                    self.bias -= learning_rate*y_label[row]

            cost = lambda_parameter*np.dot(self.weights, self.weights) + 1/m*np.sum(np.maximum(0, 1 - y_label*(np.dot(X, self.weights) - self.bias)))

            costlist.append(cost)

            if(i%(no_of_iterations/10) == 0):
                print("cost after ", i, "iteration is : ", cost)

        plt.plot(np.arange(no_of_iterations), costlist)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

    def predict(self, X) :
        return np.dot(X, self.weights) - self.bias
    



                


            