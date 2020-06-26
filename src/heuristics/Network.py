import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Flatten


class Network:
    """
    A deep convolutional neural network that takes as input the ML representation of a game (n, m, k) and tries to
    return 1 if player 1 (white in chess, red in checkers etc.) is going to win, -1 if player 2 is going to win,
    and 0 if the game is going to be a draw.
    """

    def __init__(self, input_shape, path=None):
        if path is not None:
            self.model = keras.models.load_model(path)
            if self.model.layers[0].input_shape != input_shape:
                raise Exception('input shape of loaded model doesn\'t match')
        else:
            self.model = Sequential()
            self.model.add(Conv3D(16, (3, 3, input_shape[2]), input_shape=input_shape + (1,), activation='relu'))
            self.model.add(Conv3D(16, (3, 3, input_shape[2]), activation='relu'))
            self.model.add(Flatten())
            self.model.add(Dense(8, activation='relu'))
            self.model.add(Dense(8, activation='relu'))
            self.model.add(Dense(1, activation='tanh'))
            self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def predict(self, state):
        return self.model.predict([state])

    def train(self, states, outputs):
        self.model.fit(states, outputs, epochs=1)

    def save(self, path):
        self.model.save(path)


if __name__ == '__main__':
    from src.games.Connect4 import Connect4
    print(Connect4.STATE_SHAPE)
    net = Network(Connect4.STATE_SHAPE)
    print(net)
