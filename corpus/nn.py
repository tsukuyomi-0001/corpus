from tensorflow import keras

class Network:
    def __init__(self):
        pass

    def hiddenLayers(self, units, layers: int, activation='relu'):
        self.layers = []
        if type(units) == int:
            units = [units]

        for index in range(layers):
            self.layers.append(keras.layers.Dense(units[index], activation=activation))

    def inputLayer(self, shape):
        self.input = [keras.layers.Input(shape=shape)]

        if len(shape) >= 2:
            self.input.append(keras.layers.Flatten())

    def ouputLayer(self, units, activation=None):
        self.ouput = [keras.layers.Dense(units, activation=activation)]

    def make(self):
        self.model = keras.Sequential(
            self.input +
            self.layers +
            self.ouput
        )

    def compile(self, loss: str, metrics: list, optimizer='adam'):
        self.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)