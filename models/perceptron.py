from time import sleep
import pandas as pd
import numpy as np

class Perceptron:
  def __init__(self, X : pd.DataFrame, Y : pd.DataFrame) -> None:
    self.X = X
    self.Y = Y
    self.y_i = np.zeros(len(Y))
    self.weights = np.random.random(self.X.shape[1])

  def predict(self,lr=0.1, epochs=1000):
    for _ in range(epochs):
      for i in range(self.X.shape[0]):
        x_i = self.X.iloc[i].values
        t_i = self.Y[i]
        v = np.dot(self.weights.T, x_i) + 1
        self.y_i[i] = np.sign(v)
        if(self.y_i[i] != t_i):
          L = t_i - self.y_i[i]
          self.weights = self.weights + lr * L * x_i