import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Adaline:
  def __init__(self, X : pd.DataFrame, Y : pd.DataFrame, bias=False, mse_threshold=0.1) -> None:
    self.X = X
    self.Y = Y
    self.y = np.zeros(len(Y))
    self.weights = np.random.random(self.X.shape[1])
    self.bias = bias
    self.b = np.random.random()
    self.mse_threshold = mse_threshold
    self.mse = None
  def train(self,lr=0.01, epochs=1000) -> None:
    for _ in range(epochs):
      for i in range(self.X.shape[0]):
        x_i = self.X.iloc[i].values
        t_i = self.Y.iloc[i]
        v = np.dot(self.weights.T, x_i)
        if self.bias:
          v += self.b
        self.y[i] = v
        L = t_i - self.y[i]
        self.weights = self.weights + lr * L * x_i
        self.b = self.b + lr * L
      mse = np.mean((self.y - self.Y)**2)/2
      if mse <= self.mse_threshold:
        break

  def predict(self, X : pd.DataFrame) -> np.ndarray:
    
    predictions = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
      x_i = X.iloc[i].values
      v = np.dot(self.weights.T, x_i)
      if self.bias:
        v += self.b
      predictions[i] = np.sign(v)
    return predictions