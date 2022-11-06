from models.perceptron import Perceptron
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt


class Task1:
    def __init__(self) -> None:
        self.df = pd.read_csv("data/penguins.csv")
        self.labels = list(set(self.df["species"]))
        self.features = list(set(self.df.drop(["species"], axis=1).columns))

    def run(self, config) -> None:


        # fill nulls values in gender
        self.df = self.df.fillna("Unknown")
        # species_encoder = LabelEncoder()
        # species_encoder.fit(self.df["species"])

        gender_encoder = LabelEncoder()
        gender_encoder.fit(self.df["gender"])



        ########################################

         # for dataset visualization before training

        self.df.iloc[8, 4] = 'male'
        self.df.iloc[9, 4] = 'male'
        self.df.iloc[10, 4] = 'male'

        self.df.iloc[11, 4] = 'female'
        self.df.iloc[47, 4] = 'female'
        self.df.iloc[76, 4] = 'female'

        '''
        for i in range(1, 6):
            for j in range(i + 1, 6):
                plt.figure('f1')
                fit11 = self.df.iloc[0:50, i]
                fit12 = self.df.iloc[0:50, j]

                fit21 = self.df.iloc[50:100, i]
                fit22 = self.df.iloc[50:100, j]

                fit31 = self.df.iloc[100:150, i]
                fit32 = self.df.iloc[100:150, j]

                plt.scatter(fit11, fit12)
                plt.scatter(fit21, fit22)
                plt.scatter(fit31, fit32)

                plt.xlabel(str(i))
                plt.ylabel(str(j))
                plt.show()
        '''

        ########################################

        
        classes = list(config["selected_labels"])
        class1 = classes[0]
        class2 = classes[1]

        # filter data
        df1 = self.df[self.df["species"] == class1]
        df2 = self.df[self.df["species"] == class2]

        # encode species and gender columns

        df1 = df1.assign(species=1)
        df2 = df2.assign(species=-1)

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

        cm = ConfusionMatrix(Y_test, y_pred, 1, -1)
        print("acc :", cm.accuracy())
        print("per :", cm.precision())
        print("recall :", cm.recall())
