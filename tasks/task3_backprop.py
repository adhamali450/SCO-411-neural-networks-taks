from models.network import Network
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.confusion_matrix import ConfusionMatrix
from utils.visualize import visualize

# TODO: move to seperate file and apply to task1, task2
# TODO: add other generic functions


class Task:
    # constructor
    def __init__(self, csv_path, label_col) -> None:
        self.df = pd.read_csv(csv_path)
        self.labels = list(set(self.df[label_col]))
        self.label_col = label_col
        self.features = list(set(self.df.drop([label_col], axis=1).columns))


class Task3(Task):
    def __init__(self) -> None:
        super().__init__("data/penguins.csv", "species")

    def run(self, config) -> None:
        self.df = self.df.fillna("Unknown")

        gender_encoder = LabelEncoder()
        self.df['gender'] = gender_encoder.fit_transform(self.df["gender"])

        species_encoder = OneHotEncoder()
        species = pd.DataFrame(species_encoder.fit_transform(
            self.df["species"].values.reshape(-1, 1)).toarray())

        # region train test split

        X_train = pd.concat(
            [
                self.df.drop([self.label_col], axis=1).iloc[0:30],
                self.df.drop([self.label_col], axis=1).iloc[50:80],
                self.df.drop([self.label_col], axis=1).iloc[100:130],
            ]
        )
        Y_train = pd.concat(
            [species.iloc[0:30],
             species.iloc[50:80],
             species.iloc[100:130]])

        X_test = pd.concat(
            [
                self.df.drop([self.label_col], axis=1).iloc[30:50],
                self.df.drop([self.label_col], axis=1).iloc[80:100],
                self.df.drop([self.label_col], axis=1).iloc[130:151],
            ]
        )
        Y_test = pd.concat(
            [species.iloc[30:50],
             species.iloc[80:100],
             species.iloc[130:151]])

        # endregion

        # [8, 3]: 2 layers, 8 neurons in hidden layer, 3 in the output
        # later should be specified from config (UI)
        # size=config["size"]
        self.model = Network(
            train_test_data=(X_train, Y_train, X_test, Y_test),
             size=[int(config["size"][0]),int(config["size"][1])],
              learning_rate=config["eta"],bias=config["include_bias"],
              activation=config["activation"])

        # epochs=config["epochs"]
        self.model.train(epochs=config["epochs"])

        self.model.predict()

        # cm = ConfusionMatrix(Y_test, y_pred, 1, -1)
        # print("acc :", cm.accuracy())
        # print("per :", cm.precision())
        # print("recall :", cm.recall())

        # visualize(self.p)
