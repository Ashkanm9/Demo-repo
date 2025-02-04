import numpy as np
class Linear_Regression:
    def __init__(self, lr=0.001,n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predict = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T,(y_predict - y))
            db = (1/n_samples) * np.sum(y_predict - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,new_x):
        y_predict = np.dot(new_x,self.weights) + self.bias
        return y_predict
