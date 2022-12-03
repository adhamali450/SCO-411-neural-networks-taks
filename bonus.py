import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report


class Network:
    def __init__(self, train_test_data, size, learning_rate,bias,activation) -> None:
        '''
        train_test_data: tuple of (x_train, y_train, x_test, y_test)
        size: list of layer sizes
        learning_rate: learning rate
        '''

        self.X_train, self.Y_train, self.X_test, self.Y_test = train_test_data
        self.size = size
        self.learning_rate = learning_rate
        self.bias = bias
        self.activation = activation

        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)
        self.X_test = np.array(self.X_test)
        self.Y_test = np.array(self.Y_test)

        # initialize parameters
        self.__init_params()

    def __init_params(self):
        self.params = {}


        # First layer (input layer)
        self.params[0] = {
            "A": self.X_train,
            "n": self.X_train.shape[1],
        }

        # Hidden and output layers
        for i in range(0, len(self.size)):
            self.params[i + 1] = {
                "W": np.random.randn(self.size[i], self.params[i]["n"]) * 0.01,
                "b": np.zeros((1, self.size[i])),
                "n": self.size[i]

                # Z, A calculated later in forward prop
                # dA, dZ, dW, db calculated later in backward prop
            }

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __sigmoid_derivative(self, z):
        sig = self.__sigmoid(z)
        return sig * (1 - sig)

    def __tanh_derivative(self, z):
        return 1 - np.power(np.tanh(z), 2)

    def __mse(self, y, y_hat):
        return np.mean(np.square(y - y_hat))

    def prop_forward(self):
        for i in range(1, len(self.size) + 1):
            # Z = W * A + b
            self.params[i]["Z"] = np.dot(self.params[i - 1]["A"],self.params[i]["W"].T) 
            if self.bias:
                self.params[i]["Z"] +=  self.params[i]["b"]

            # A = tanh(Z)
            if self.activation == "tanh" :
                self.params[i]["A"] = np.tanh(self.params[i]["Z"])
            else:
                self.params[i]["A"] = self.__sigmoid(self.params[i]["Z"])

    def prop_backward(self):
        #######################
        # handle output layer #
        #######################

        # dZ = A - Y
        out_index = len(self.size)

        self.params[out_index]["dZ"] = self.params[out_index]["A"] - self.Y_train


        ########################
        # handle hidden layers #
        ########################

        for i in range(out_index, 0, -1):
            # dZ = np.dot(W_next.T, dZ_next) * g'(Z)
            if self.activation == "tanh" :
                self.params[i]["dZ"] *= self.__tanh_derivative(self.params[i]["Z"])
            else :
                self.params[i]["dZ"] *= self.__sigmoid_derivative(self.params[i]["Z"])

        # dW = np.dot(dZ, A_prev.T) / m
            m = self.X_train.shape[0]

            self.params[i]["dW"] = np.dot(self.params[i - 1]["A"].T,self.params[i]["dZ"]) / m

            self.params[i-1]["dZ"] = np.dot(self.params[i]["dZ"],self.params[i]["W"])

            # db = np.sum(dZ, axis=1, keepdims=True) / m
            if self.bias:
                self.params[i]["db"] = np.sum(self.params[i]["dZ"],axis=0,keepdims=True) / m

    def update_params(self):
        for i in range(1, len(self.size) + 1):
            self.params[i]["W"] = self.params[i]["W"] - self.learning_rate * self.params[i]["dW"].T
            if self.bias:
                self.params[i]["b"] = self.params[i]["b"] - self.learning_rate * self.params[i]["db"]

    def train(self, epochs):
        for _ in range(epochs):
            self.prop_forward()
            self.prop_backward()
            self.update_params()



    def predict(self):
        

        A_temp = self.X_test
        Z_temp = 0
        predictions = np.zeros((self.X_test.shape[0],self.X_test.shape[1]))

        for i in range(1, len(self.size) + 1):

            Z_temp = np.dot(A_temp,self.params[i]["W"].T) 

            if self.bias:
                Z_temp += self.params[i]["b"]

            # A = tanh(Z)
            if self.activation == "tanh" :
                    A_temp= np.tanh(Z_temp)
            else:
                A_temp = self.__sigmoid(Z_temp)

        predictions = A_temp
        output = np.ndarray((predictions.shape))

        
        for i in range(predictions.shape[0]):
            max = np.argmax(predictions[i])
            # print(i,max)
            if max == 0:
                output[i][0] = 1
                output[i][1] = 0
                output[i][2] = 0
            elif max == 1:
                output[i][1] = 1
                output[i][0] = 0
                output[i][2] = 0
            else :
                output[i][2] = 1
                output[i][0] = 0
                output[i][1] = 0
        return output

df_train = pd.read_csv('./data/mnist_train.csv')
df_test = pd.read_csv('./data/mnist_test.csv')
labels = list(set(df_train['label']))

scaler = MaxAbsScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)


X_train = df_train.drop(['label'], axis=1)
Y_train = np.array(df_train['label']).reshape(-1, 1)


X_test = df_test.drop(['label'], axis=1)
Y_test = np.array(df_test['label']).reshape(-1,1)

network =  Network((X_train, Y_train, X_test, Y_test), [20, 15, 10], 0.001, True, 'sigmoid')
network.train(100)

Y_predict = network.predict()

accuracy = 0
for i in range(len(Y_predict)):
    if np.argmax(Y_predict[i]) == np.argmax(Y_test[i, :]):
        accuracy += 1
print(f'Accuracy: {accuracy / len(Y_predict)}')