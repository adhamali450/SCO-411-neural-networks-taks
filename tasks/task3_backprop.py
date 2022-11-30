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
        self.features = list(set(self.df.drop([label_col], axis=1).columns))


class Task3(Task):
    def __init__(self) -> None:
        super().__init__("data/penguins.csv", "species")

    def run(self, config) -> None:
        self.df = self.df.fillna("Unknown")
        gender_encoder = LabelEncoder()
        gender_encoder.fit(self.df["gender"])

        species_encoder = OneHotEncoder()
        species = pd.DataFrame(species_encoder.fit_transform(
            self.df["species"].values.reshape(-1, 1)).toarray())

        # region train test split

        X_train = pd.concat(
            [
                self.df.drop([self.label_col], axis=1).iloc[1:30],
                self.df.drop([self.label_col], axis=1).iloc[50:80],
                self.df.drop([self.label_col], axis=1).iloc[100:130],
            ]
        )
        Y_train = pd.concat(
            [species.iloc[1:30],
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
             species.iloc[80:100]])

        # endregion

        # [8, 3]: 2 layers, 8 neurons in hidden layer, 3 in the output
        # later should be specified from config (UI)
        # size=config["size"]
        self.model = Network(
            train_test_data=(X_train, Y_train, X_test, Y_test), size=[8, 3], learning_rate=0.0001)

        # epochs=config["epochs"]
        self.model.train(epochs=1000)

        y_pred = self.model.predict(X_test)

        cm = ConfusionMatrix(Y_test, y_pred, 1, -1)
        print("acc :", cm.accuracy())
        print("per :", cm.precision())
        print("recall :", cm.recall())

        visualize(self.p)