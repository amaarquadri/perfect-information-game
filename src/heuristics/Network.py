import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Flatten
from multiprocessing import Process, Pipe


class Network:
    """
    A deep convolutional neural network that takes as input the ML representation of a game (n, m, k) and tries to
    return 1 if player 1 (white in chess, red in checkers etc.) is going to win, -1 if player 2 is going to win,
    and 0 if the game is going to be a draw.

    Network will receive all inputs and output moves in the format of flattened arrays of legal moves. \
    It will internally handle conversion too and from move distribution matrices using GameClass.get_legal_moves
    """

    def __init__(self, GameClass, policy_path=None, evaluation_path=None):
        self.GameClass = GameClass
        self.policy_path = policy_path
        self.evaluation_path = evaluation_path

        # lazily initialized so Network can be passed between processes before being initialized
        self.policy_model = None
        self.evaluation_model = None

    def initialize(self):
        if self.policy_model is not None and self.evaluation_model is not None:
            return

        input_shape = self.GameClass.STATE_SHAPE
        output_shape = self.GameClass.MOVE_SHAPE
        output_neurons = np.product(output_shape)

        if self.policy_path is not None:
            self.policy_model = keras.models.load_model(self.policy_path)
            if self.policy_model.input_shape != (None,) + input_shape + (1,):
                raise Exception('input shape of loaded model doesn\'t match')
        else:
            self.policy_model = Sequential()
            self.policy_model.add(Conv3D(32, (3, 3, 1), input_shape=input_shape + (1,), activation='relu'))
            self.policy_model.add(Conv3D(32, (3, 3, input_shape[2] - 1), activation='relu'))
            self.policy_model.add(Flatten())
            self.policy_model.add(Dense(16, activation='relu'))
            self.policy_model.add(Dense(16, activation='relu'))
            self.policy_model.add(Dense(output_neurons, activation='softmax'))
            self.policy_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if self.evaluation_path is not None:
            self.evaluation_model = keras.models.load_model(self.evaluation_path)
            if self.evaluation_model.input_shape != (None,) + input_shape + (1,):
                raise Exception('input shape of loaded model doesn\'t match')
        else:
            self.evaluation_model = Sequential()
            self.evaluation_model.add(Conv3D(32, (3, 3, 1), input_shape=input_shape + (1,), activation='relu'))
            self.evaluation_model.add(Conv3D(32, (3, 3, input_shape[2] - 1), activation='relu'))
            self.evaluation_model.add(Flatten())
            self.evaluation_model.add(Dense(16, activation='relu'))
            self.evaluation_model.add(Dense(16, activation='relu'))
            self.evaluation_model.add(Dense(1, activation='tanh'))
            self.evaluation_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def choose_move(self, position):
        distribution = self.policy(position)[self.GameClass.get_legal_moves(position) == 1]
        distribution = distribution / np.sum(distribution)

        idx = np.random.choice(np.arange(len(distribution)), p=distribution)
        return self.GameClass.get_possible_moves(position)[idx]

    def policy(self, state):
        raw_policy = np.squeeze(self.policy_model.predict(state[np.newaxis, :, :, :, np.newaxis]))

        filtered_policy = raw_policy[self.GameClass.get_legal_moves(state) == 1]
        filtered_policy = filtered_policy / np.sum(filtered_policy)
        return filtered_policy

    def evaluation(self, state):
        return np.squeeze(self.evaluation_model.predict(state[np.newaxis, :, :, :, np.newaxis]))

    def train(self, data):
        states = []
        policy_outputs = []
        value_outputs = []

        for game, outcome in data:
            for position, distribution in game:
                if type(distribution) == list:
                    continue
                legal_moves = self.GameClass.get_legal_moves(position)
                policy = np.zeros_like(legal_moves)
                policy[legal_moves == 1] = distribution

                states.append(position)
                # policy outputs are flattened to match policy output's flat dense output layer
                policy_outputs.append(policy.flatten())
                value_outputs.append(outcome)

        input_data = np.stack(states, axis=0)[:, :, :, :, np.newaxis]
        policy_outputs = np.stack(policy_outputs, axis=0)
        value_outputs = np.array(value_outputs)
        print('Samples:', input_data.shape[0])

        shuffle = np.arange(input_data.shape[0])
        np.random.shuffle(shuffle)
        input_data = input_data[shuffle, ...]
        policy_outputs = policy_outputs[shuffle, ...]
        value_outputs = value_outputs[shuffle, ...]

        self.policy_model.fit(input_data, policy_outputs, epochs=100)
        self.evaluation_model.fit(input_data, value_outputs, epochs=100)

    def save(self, policy_path, evaluation_path):
        self.policy_model.save(policy_path)
        self.evaluation_model.save(evaluation_path)

    @staticmethod
    def spawn_process(GameClass, policy_path=None, evaluation_path=None, pipes=1):
        parent_policy_pipes, child_policy_pipes = zip(*[Pipe() for _ in range(pipes)])
        parent_evaluation_pipes, child_evaluation_pipes = zip(*[Pipe() for _ in range(pipes)])
        process = Process(target=Network.process_loop, args=(GameClass, policy_path, evaluation_path,
                                                             parent_policy_pipes, parent_evaluation_pipes))

        proxy_networks = [ProxyNetwork(policy_pipe, evaluation_pipe)
                          for policy_pipe, evaluation_pipe in zip(child_policy_pipes, child_evaluation_pipes)]
        return process, proxy_networks

    @staticmethod
    def process_loop(GameClass, policy_path, evaluation_path, policy_pipes, evaluation_pipes):
        network = Network(GameClass, policy_path, evaluation_path)
        network.initialize()

        with tf.device('/GPU:0'):
            while True:
                for policy_pipe in policy_pipes:
                    if policy_pipe.poll():
                        position = policy_pipe.recv()
                        filtered_distribution = network.policy(position)
                        policy_pipe.send(filtered_distribution)
                for evaluation_pipe in evaluation_pipes:
                    if evaluation_pipe.poll():
                        position = evaluation_pipe.recv()
                        evaluation = network.evaluation(position)
                        evaluation_pipe.send(evaluation)


class ProxyNetwork:
    def __init__(self, policy_pipe, evaluation_pipe):
        self.policy_pipe = policy_pipe
        self.evaluation_pipe = evaluation_pipe

    def initialize(self):
        """
        Added so that a ProxyNetwork can be used in place of a regular Network with no issues when initialize is called.
        """
        pass

    def policy(self, state):
        self.policy_pipe.send(state)
        return self.policy_pipe.recv()

    def evaluation(self, state):
        self.evaluation_pipe.send(state)
        return self.evaluation_pipe.recv()


def train():
    from src.games.Connect4 import Connect4
    import os
    import pickle
    net = Network(Connect4, '../heuristics/models/policy0.h5', '../heuristics/models/evaluation0.h5')
    net.initialize()
    print('Policy network size: ', net.policy_model.count_params())
    print('Evaluation network size: ', net.evaluation_model.count_params())

    data = []
    for file in os.listdir('../heuristics/games/mcts_network0_games'):
        with open(f'../heuristics/games/mcts_network0_games/{file}', 'rb') as fin:
            data.append(pickle.load(fin))
    net.train(data)
    net.save('../heuristics/models/policy_test.h5', '../heuristics/models/evaluation_test.h5')


if __name__ == '__main__':
    train()
