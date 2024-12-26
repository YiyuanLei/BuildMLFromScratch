import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.001, epoch = 1000):
        self.lr  = lr
        self.num_iterations = epoch
        self.weights = None
        self.bias = None

    def cross_entropy_loss(self, y, y_hat):
        return -(y * np.log(y_hat) + (1-y)*np.log(1-y_hat)).mean()
    def sigmoid(self, x):
        return np.exp(x)/ (1 +  np.exp(x))
    def fit(self, x, y):
        num_samples, num_features = x.shape()
        self.weights, self.bias = np.zeros(num_features), 0

        for _ in range(self.num_iterations):
            prediction = self.sigmoid(x.T.dot(self.weights) + self.bias) # (n, 1)
            error = prediction - y # (n, 1)
            d_weights = x.T.dot(error)/num_samples
            d_bias = error.sum()/num_samples

            self.weights -= self.lr *  d_weights
            self.bias -= self.lr * d_bias

        return self.weights,self.b
    def predict(self, x):
        prediction = self.sigmoid(np.dot(x.T, self.weights) + self.bias)
        return prediction
