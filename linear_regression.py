# Linear Regression :
# Matrix Representation (MathisPower4u) : https://www.youtube.com/watch?v=Qa_FI92_qo8
# Formula Derivation (towardsdatascience) : https://towardsdatascience.com/the-matrix-algebra-of-linear-regression-6fb433f522d5

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


class LinearRegression :
    def __init__(self, x, y):
        self._X = self._convertX(x)
        self._Y = y
        self.weights = self._fit()

    def _convertX(self, x):
        # [x1, y1, z1] -> [1, x1, y1, z1]
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        
    def _fit(self):
        temp1 = np.dot(self._X.T, self._X)
        temp2 = np.dot(self._X.T, self._Y)

        return np.dot(np.linalg.inv(temp1), temp2)
    
    def predict(self, x):
        return np.dot(self._convertX(x), self.weights)
    
    def mean_squared_error(self, x, y):
        y_actual = y
        y_predicted = self.predict(x)

        E = y_actual - y_predicted

        return np.dot(E.T, E) / len(E)



diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

# My model
myModel = LinearRegression(diabetes_X_train, diabetes_Y_train)
# scikit learn model
skModel = linear_model.LinearRegression()
skModel.fit(diabetes_X_train, diabetes_Y_train)

# Predictions
print("My model:  ", myModel.predict(diabetes_X_test))
print("Scikit learn model:  ", skModel.predict(diabetes_X_test))

# Error
print("My model:  ", myModel.mean_squared_error(diabetes_X_test, diabetes_Y_test))
print("Scikit learn model:  ", mean_squared_error(diabetes_Y_test, skModel.predict(diabetes_X_test)))

# Weights
print("My model:  ", myModel.weights)
print("Scikit learn model:  ", skModel.coef_)
