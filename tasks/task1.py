from models.perceptron import Perceptron
import pandas as pd
from sklearn.model_selection import train_test_split



def task1():
  df = pd.read_csv('data/penguins.csv')

  #fill nulls values in gender
  df = df.fillna("Unknown")
  
  #spilt samples into 3 df of 50
  df1 = df.iloc[:50]
  df2 = df.iloc[50:100]
  df3 = df.iloc[100:]

  X1 = df1.drop(['species'],axis=1)
  Y1 = df1['species']


  X2 = df2.drop(['species'],axis=1)
  Y2 = df2['species']
  

  X3 = df3.drop(['species'],axis=1)
  Y3 = df3['species'] 
  

  #train test split
  x_train1, x_test1, y_true_train1, y_true_test1 = train_test_split(X1, Y1, test_size=0.3, shuffle=True, random_state=10)

  x_train2, x_test2, y_true_train2, y_true_test2 = train_test_split(X2, Y2, test_size=0.3, shuffle=True, random_state=10)

  x_train3, x_test3, y_true_train3, y_true_test3 = train_test_split(X3, Y3, test_size=0.3, shuffle=True, random_state=10)



  

  #print(df.info())
  #print(df[df.isna().any(axis=1)])
  

  # X = df.drop(['species', 'gender'], axis=1)
  # Y = df['species']
  # encoder = LabelEncoder()
  # Y = encoder.fit_transform(Y)



  p = Perceptron(X, Y)

  p.predict()

  print(p.y_i)