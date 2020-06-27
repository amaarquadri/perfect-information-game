import numpy as np
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
            self.model.add(Conv3D(32, (3, 3, 1), input_shape=input_shape + (1,), activation='relu'))
            self.model.add(Conv3D(32, (3, 3, input_shape[2] - 1), activation='relu'))
            self.model.add(Flatten())
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dense(7, activation='softmax'))
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, state):
        return self.model.predict([state])

    def train(self, data):
        states = []
        outputs = []
        for game, outcome in data:
            for position, distribution in game:
                states.append(position)
                outputs.append(distribution)

        self.model.fit(np.stack(states, axis=0), np.stack(outputs, axis=0), epochs=1)

    def save(self, path):
        self.model.save(path)
