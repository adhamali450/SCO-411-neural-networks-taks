from time import sleep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
  def __init__(self, X : pd.DataFrame, Y : pd.DataFrame, bias=False) -> None:
    self.X = X
    self.Y = Y
    self.y_i = np.zeros(len(Y))
    self.weights = np.random.random(self.X.shape[1])
    self.bias = bias
    self.b = np.random.random()


  def train(self,lr=0.01, epochs=1000) -> None:
    for _ in range(epochs):
      for i in range(self.X.shape[0]):
        x_i = self.X.iloc[i].values
        t_i = self.Y.iloc[i]
        v = np.dot(self.weights.T, x_i)
        if self.bias:
          v += self.b
        self.y_i[i] = np.sign(v) 
        if(self.y_i[i] != t_i):
          L = t_i - self.y_i[i]
          self.weights = self.weights + lr * L * x_i
          self.b = self.b + lr * L 

    ########################################

    # for dataset visualization after training
    
    if self.bias == False:
      plotbias=0
    else:
      plotbias = self.b

    plt.figure('f1')
    plotbias=self.b

    fit11 = self.X.iloc[0:30, 0]
    fit12 = self.X.iloc[0:30, 1]

    fit21 = self.X.iloc[30:60, 0]
    fit22 = self.X.iloc[30:60, 1]

    plt.scatter(fit11, fit12)
    plt.scatter(fit21, fit22)

    x1Valus=[]
    x2Values=[]

    x1Valus.append(self.X.iloc[0, 0])
    x1Valus.append(self.X.iloc[1, 0])
    x1Valus.append(self.X.iloc[2, 0])
    x1Valus.append(50)

    x1Valus.append(self.X.iloc[40, 0])
    x1Valus.append(self.X.iloc[41, 0])
    x1Valus.append(self.X.iloc[42, 0])

    #if self.bias != False:
    #  x1Valus.append(0)

    for i in x1Valus:
      x2Values.append(((-plotbias) - (self.weights[0] * i))/self.weights[1])

    plt.plot(x1Valus, x2Values)
    colNames=self.X.columns.tolist()
    plt.xlabel(str(colNames[0]))
    plt.ylabel(str(colNames[1]))
    plt.show()

    ########################################

  def predict(self, X : pd.DataFrame) -> np.ndarray:
    
    predictions = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
      x_i = X.iloc[i].values
      v = np.dot(self.weights.T, x_i)
      if self.bias:
        v += self.b
      predictions[i] = np.sign(v)
    return predictions