import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, epoch):
        self.learning_rate  = learning_rate
        self.epoch = epoch

    def sigmoid(self, x):
        return np.exp(x)/ (1 +  np.exp(x))
    def fit(self, x, y):
        num_samples, num_features = x.shape()
        self.W, self.b = np.zeros(num_features, 1), np.zeros(num_samples, 1)

        for _ in range(self.epoch):
            f_x = np.dot(x.T, self.W) + self.b # (n, 1)
            h = self.sigmoid(f_x) # (n, 1)
            dW = (1/num_features) * np.dot(y, (x -f_x))
            db = (1/num_features) * np.sum (h - y)

            self.W = +- self.learning_rate * dW
            self.b = +- self.learning_rate * db

        return self.W,self.b
    def predict(self, x):
        prediction = self.sigmoid(np.dot(x.T, self.W) + self.b)
        return prediction
