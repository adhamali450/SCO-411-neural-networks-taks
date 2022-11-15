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

    def visualize(self) -> None:
        ########################################

        # for dataset visualization after training
        
        if self.p.bias == False:
            plotbias=0
        else:
            plotbias = self.p.b

        plt.figure('f1')
        plotbias=self.p.b

        fit11 = self.p.X.iloc[0:30, 0]
        fit12 = self.p.X.iloc[0:30, 1]

        fit21 = self.p.X.iloc[30:60, 0]
        fit22 = self.p.X.iloc[30:60, 1]

        plt.scatter(fit11, fit12)
        plt.scatter(fit21, fit22)

        x1Valus=[]
        x2Values=[]

        x1Valus.append(self.p.X.iloc[0, 0])
        x1Valus.append(self.p.X.iloc[1, 0])
        x1Valus.append(self.p.X.iloc[2, 0])
        x1Valus.append(50)

        x1Valus.append(self.p.X.iloc[40, 0])
        x1Valus.append(self.p.X.iloc[41, 0])
        x1Valus.append(self.p.X.iloc[42, 0])


        for i in x1Valus:
            x2Values.append(((-plotbias) - (self.p.weights[0] * i))/self.p.weights[1])

        plt.plot(x1Valus, x2Values)
        colNames=self.p.X.columns.tolist()
        plt.xlabel(str(colNames[0]))
        plt.ylabel(str(colNames[1]))
        plt.show()

        ########################################
    def run(self, config) -> None:
        self.df = self.df.fillna("Unknown")

        gender_encoder = LabelEncoder()
        gender_encoder.fit(self.df["gender"])

        # region preparing data and preprocessing

        classes = list(config["selected_labels"].values())
        class1 = classes[0]
        class2 = classes[1]

        # filter data
        df1 = self.df[self.df["species"] == class1]
        df2 = self.df[self.df["species"] == class2]

        # encode species and gender columns
        # 1, -1 for the signum function
        df1 = df1.assign(species=1)
        df2 = df2.assign(species=-1)

        df1["gender"] = gender_encoder.transform(df1["gender"])
        df2["gender"] = gender_encoder.transform(df2["gender"])

        # filter features
        features = list(config["selected_features"].values())
        features.append("species")
        df1 = df1[features]
        df2 = df2[features]

        # endregion

        X_train = pd.concat(
            [
                df1.drop(["species"], axis=1).iloc[:30],
                df2.drop(["species"], axis=1).iloc[:30],
            ]
        )
        Y_train = pd.concat(
            [df1["species"].iloc[:30], df2["species"].iloc[:30]])

        X_test = pd.concat(
            [
                df1.drop(["species"], axis=1).iloc[30:],
                df2.drop(["species"], axis=1).iloc[30:],
            ]
        )
        Y_test = pd.concat(
            [df1["species"].iloc[30:], df2["species"].iloc[30:]])

        self.p = Perceptron(X_train, Y_train, bias=config["include_bias"])

        self.p.train(lr=config["eta"], epochs=config["epochs"])

        y_pred = self.p.predict(X_test)

        cm = ConfusionMatrix(Y_test, y_pred, 1, -1)
        print("acc :", cm.accuracy())
        print("per :", cm.precision())
        print("recall :", cm.recall())
        self.visualize()
