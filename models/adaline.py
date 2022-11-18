import pandas as pd
import numpy as np


class Adaline:
    def __init__(self, X, Y, bias=False, mse_threshold=0.1) -> None:
        self.X = X
        self.Y = Y
        self.y = np.zeros(len(Y))
        self.weights = np.random.random(self.X.shape[1])
        self.bias = bias
        self.b = np.random.random()
        self.mse_threshold = mse_threshold
        self.mse = None

    def train(self, lr=0.01, epochs=1000) -> None:
        try:
            X = self.X.to_numpy(np.float32)
            Y = self.Y.to_numpy(np.float32)
        except:
            pass
        for _ in range(epochs):
            for i in range(self.X.shape[0]):
                x_i = X[i]
                t_i = Y[i]
                v = np.dot(self.weights.T, x_i)
                if self.bias:
                    v += self.b
                self.y[i] = v
                L = t_i - self.y[i]
                self.weights = self.weights + lr * L * x_i
                if self.bias:
                    self.b = self.b + lr * L
            self.mse = np.mean((self.y - self.Y) ** 2) / 2
            if self.mse <= self.mse_threshold:
                break

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            X = X.to_numpy(np.float32)
        except:
            pass
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            v = np.dot(self.weights.T, X[i])
            if self.bias:
                v += self.b
            predictions[i] = np.sign(v)
        return predictions
