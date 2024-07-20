# K Nearest Neighbours (KNN)
# Basic Idea (Code With Harry) : https://www.youtube.com/watch?v=P4KD9GhA15Y

import numpy as np
from sklearn import datasets, neighbors


class KNeighborsClassifier:
    def __init__(self, k=3) -> None:
        self._Y = None
        self._X = None
        self.k = k

    def fit(self, X, Y):
        self._X = X
        self._Y = Y

    def predict(self, X):
        Y = []

        for i, e in enumerate(X):
            distances = np.linalg.norm(self._X - e, axis=1)
            nearest_neighbors_indices = np.argpartition(distances, self.k)[:self.k]

            nearest_neighbors_labels = self._Y[nearest_neighbors_indices]

            out = np.argmax(np.bincount(nearest_neighbors_labels))
            Y.append(out)

        return Y


iris = datasets.load_iris()
features = iris.data
label = iris.target

# clf = neighbors.KNeighborsClassifier(3)
clf = KNeighborsClassifier(3)
clf.fit(features, label)

predictions = clf.predict([[1, 1, 1, 1], [5, 5, 5, 5]])

print(predictions)
