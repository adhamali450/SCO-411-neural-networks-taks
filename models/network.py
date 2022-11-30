import numpy as np

# TODO: Add support to no bias
# TODO: Add support to specify activation function (currently only tanh)


class Network:
    def __init__(self, train_test_data, size, learning_rate) -> None:
        '''
        train_test_data: tuple of (x_train, y_train, x_test, y_test)
        size: list of layer sizes
        learning_rate: learning rate
        '''
        self.X_train, self.Y_train, self.X_test, self.Y_test = train_test_data
        self.size = size
        self.learning_rate = learning_rate

        # initialize parameters
        self.__init_params()

    def __init_params(self):
        self.params = {}

        # First layer (input layer)
        self.params[0] = {
            "A": self.X_train
        }

        # Hidden and output layers
        for i in range(1, len(self.size) + 1):
            self.params[i] = {
                "W": np.random.randn(self.size[i], self.size[i - 1]),
                "b": np.zeros((self.size[i], 1)),

                # Z, A calculated later in forward prop
                # dA, dZ, dW, db calculated later in backward prop
            }

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def prop_forward(self):
        for i in len(1, self.size + 1):
            # Z = W * A + b
            self.params[i]["Z"] = \
                np.dot(self.params[str(i)]["W"], self.params[i - 1]["A"]) \
                + self.params[i]["b"]

            # For hidden layers: A = tanh(Z)
            # For output layer: A = sigmoid(Z)
            if i < len(self.size):
                self.params[i]["A"] = np.tanh(self.params[i]["Z"])
            else:
                self.params[i]["A"] = self.__sigmoid(self.params[i]["Z"])

    def prop_backward(self):
        # handle latest layer
        # dZ = A - Y
        self.params[len(self.size)]["dZ"] = self.params[len(
            self.size)]["A"] - self.Y_train

        # dW = np.dot(dZ, A_prev.T) / m
        m = self.X_train.shape[1]
        self.params[len(self.size)]["dW"] = np.dot(
            self.params[len(self.size)]["dZ"], self.params[len(self.size) - 1]["A"].T) / m

        # db = np.sum(dZ, axis=1, keepdims=True) / m
        self.params[len(self.size)]["db"] = np.sum(
            self.params[len(self.size)]["dZ"], axis=1, keepdims=True) / m

        # handle hidden layers
        for i in range(len(self.size) - 1, 1, -1):
            # dZ = np.dot(W_next.T, dZ_next) * (1 - np.power(A, 2))
            self.params[i]["dZ"] = \
                np.dot(
                    self.params[i + 1]["W"].T,
                    self.params[i + 1]["dZ"]) * \
                (1 - np.power(self.params[i]["A"], 2))
            # dW = np.dot(dZ, A_prev.T) / m
            self.params[i]["dW"] = np.dot(
                self.params[i]["dZ"], self.params[i - 1]["A"].T) / m
            # db = np.sum(dZ, axis=1, keepdims=True) / m
            self.params[i]["db"] = np.sum(
                self.params[i]["dZ"], axis=1, keepdims=True) / m

    def update_params(self):
        for i in range(1, len(self.size) + 1):
            self.params[i]["W"] -= self.learning_rate * self.params[i]["dW"]
            self.params[i]["b"] -= self.learning_rate * self.params[i]["db"]

    def train(self, epochs):
        self.prop_forward()

        for _ in range(epochs):
            self.prop_backward()
            self.update_params()

    def predict(self, x):
        pass
