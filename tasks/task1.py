from models.perceptron import Perceptron
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.confusion_matrix import ConfusionMatrix
from utils.visualize import visualize


class Task1:
    def __init__(self) -> None:
        self.df = pd.read_csv("data/penguins.csv")
        self.labels = list(set(self.df["species"]))
        self.features = list(set(self.df.drop(["species"], axis=1).columns))

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

        # region train_test split

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

        # endregion

        self.p = Perceptron(X_train, Y_train, bias=config["include_bias"])

        self.p.train(lr=config["eta"], epochs=config["epochs"])

        y_pred = self.p.predict(X_test)

        cm = ConfusionMatrix(Y_test, y_pred, 1, -1)
        print("acc :", cm.accuracy())
        print("per :", cm.precision())
        print("recall :", cm.recall())
        visualize(self.p)
