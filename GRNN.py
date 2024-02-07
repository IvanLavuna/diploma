import numpy as np

class GRNN:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None

    def _kernel_function(self, x, x_train):
        return np.exp(-np.sum((x - x_train)**2) / (2 * self.sigma**2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            weighted_sum = 0
            sum_of_weights = 0
            for x_train, y_train in zip(self.X_train, self.y_train):
                kernel_value = self._kernel_function(x_test, x_train)
                weighted_sum += kernel_value * y_train
                sum_of_weights += kernel_value
            prediction = weighted_sum / sum_of_weights if sum_of_weights != 0 else 0
            predictions.append(prediction)
        return np.array(predictions)