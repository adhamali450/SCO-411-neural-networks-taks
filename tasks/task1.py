from models.perceptron import Perceptron
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def task1():
  df = pd.read_csv('data/penguins.csv')
  
  # print(df.info())
  # print(df[df.isna().any(axis=1)])

  X = df.drop(['species', 'gender'], axis=1)
  Y = df['species']
  encoder = LabelEncoder()
  Y = encoder.fit_transform(Y)
  p = Perceptron(X, Y)

  p.predict()

  print(p.y_i)