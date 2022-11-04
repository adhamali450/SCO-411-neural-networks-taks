from models.perceptron import Perceptron
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
from utils.confusion_matrix import ConfusionMatrix
pd.options.mode.chained_assignment = None  # default='warn'




class Task1:
    def __init__(self) -> None:
        self.df = pd.read_csv("data/penguins.csv")
        self.labels = list(set(self.df["species"]))
        self.features = list(set(self.df.drop(["species"], axis=1).columns))

    def run(self, config) -> None:

        #TODO : visualizations for report
        #Write visualizations code here
      

        # fill nulls values in gender
        self.df = self.df.fillna("Unknown")
        # species_encoder = LabelEncoder()
        # species_encoder.fit(self.df["species"])

        gender_encoder = LabelEncoder()
        gender_encoder.fit(self.df["gender"])

        classes = list(config["selected_labels"])
        class1 = classes[0]
        class2 = classes[1]

        # filter data
        df1 = self.df[self.df["species"] == class1]
        df2 = self.df[self.df["species"] == class2]

        


        # encode species and gender columns

        df1["species"] = df1["species"].replace(class1 , 1)
        df2["species"] = df2["species"].replace(class2 , -1)


        df1["gender"] = gender_encoder.transform(df1["gender"])
        df2["gender"] = gender_encoder.transform(df2["gender"])

        # filter features
        features = list(config["selected_features"])
        features.append("species")
        df1 = df1[features]
        df2 = df2[features]

        X_train = pd.concat(
            [
                df1.drop(["species"], axis=1).iloc[:30],
                df2.drop(["species"], axis=1).iloc[:30],
            ]
        )
        Y_train = pd.concat([df1["species"].iloc[:30], df2["species"].iloc[:30]])

        X_test = pd.concat(
            [
                df1.drop(["species"], axis=1).iloc[30:],
                df2.drop(["species"], axis=1).iloc[30:],
            ]
        )
        Y_test = pd.concat([df1["species"].iloc[30:], df2["species"].iloc[30:]])

        

        p = Perceptron(X_train, Y_train, bias=config["include_bias"])

        p.train(lr=config["eta"], epochs=config["epochs"])

        y_pred = p.predict(X_test)
       

        cm = ConfusionMatrix(Y_test,y_pred,1,-1)
        print("acc :" , cm.accuracy())
        print("per :" , cm.precision())
        print("recall :" , cm.recall())

