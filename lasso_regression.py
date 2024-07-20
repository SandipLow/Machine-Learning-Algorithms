import numpy as np


class LASSO:
    def __init__(self):
        self.weights = None
        self.bias = None

    # fitting the dataset to LASSO regression
    def fit(self, X, Y, learning_rate, no_of_iterations, lambda_parameter):
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        # Gradient Descent
        for i in range(no_of_iterations):
            # linear equation of the model
            Y_prediction = self.predict(X)
            dW = np.zeros(n)

            for i in range(n):

                if self.weights[i] > 0:
                    dW[i] = (-2 / m) * (X[:, i].dot(Y - Y_prediction) + lambda_parameter)

                else:
                    dW[i] = (-2 / m) * (X[:, i].dot(Y - Y_prediction) - lambda_parameter)

            dB = (-2 / m) * (Y - Y_prediction)

            # update weights and bias
            self.weights -= learning_rate * dW
            self.bias -= learning_rate * dB

    # predicting the target variables
    def predict(self, X):

        return np.dot(X, self.weights) + self.bias

# Testing the model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# loading the dataset
X, Y = datasets.load_diabetes(return_X_y=True)

# splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# scaling the dataset
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# training the model
model = LASSO()
model.fit(X_train, Y_train, 0.01, 1000, 0.01)

# predicting the target variables
Y_prediction = model.predict(X_test)

# calculating the mean squared error
mse = mean_squared_error(Y_test, Y_prediction)
print("Mean Squared Error :", mse)

# plotting the results
plt.scatter(Y_test, Y_prediction)
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs Predicted")
plt.show()


