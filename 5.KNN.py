import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        return np.array([Counter(self.y_train[np.argsort([np.linalg.norm(x - x_train) for x_train in self.X_train])[:self.k]]).most_common(1)[0][0] for x in X_test])

# Example usage
if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 3], [6, 6]])

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print(predictions)  # Output: [0 1]
