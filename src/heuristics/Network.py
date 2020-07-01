from time import time
from multiprocessing import Process, Pipe
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Flatten
from keras.callbacks import TensorBoard


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
            self.evaluation_model.add(Conv3D(8, (3, 3, 1), input_shape=input_shape + (1,), activation='relu'))
            # self.evaluation_model.add(Conv3D(16, (3, 3, input_shape[2] - 1), activation='relu'))
            self.evaluation_model.add(Flatten())
            # self.evaluation_model.add(Dense(8, activation='relu'))
            # self.evaluation_model.add(Dense(16, activation='relu'))
            self.evaluation_model.add(Dense(1, activation='tanh'))
            self.evaluation_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    def choose_move(self, position):
        distribution = self.policy(position)
        idx = np.random.choice(np.arange(len(distribution)), p=distribution)
        return self.GameClass.get_possible_moves(position)[idx]

    def policy(self, state):
        raw_policy = np.squeeze(self.policy_model.predict(state[np.newaxis, :, :, :, np.newaxis]))

        filtered_policy = raw_policy[self.GameClass.get_legal_moves(state)]
        filtered_policy = filtered_policy / np.sum(filtered_policy)
        return filtered_policy

    def evaluation(self, state):
        return np.squeeze(self.evaluation_model.predict(state[np.newaxis, :, :, :, np.newaxis]))

    def train(self, data, validation_fraction=0.2):
        split = int((1 - validation_fraction) * len(data))
        train_input, train_policy, train_value = self.process_data(data[:split])
        test_input, test_policy, test_value = self.process_data(data[split:])
        print('Training Samples:', train_input.shape[0])
        print('Validation Samples:', test_input.shape[0])

        # self.policy_model.fit(train_input, train_policy, epochs=20, validation_data=(test_input, test_policy),
        #                       callbacks=[TensorBoard(log_dir=f'../heuristics/logs/policy_model-{time()}')])
        self.evaluation_model.fit(train_input, train_value, epochs=20, validation_data=(test_input, test_value),
                                  callbacks=[TensorBoard(log_dir=f'../heuristics/logs/evaluation_model-{time()}')])

    def process_data(self, data):
        states = []
        policy_outputs = []
        value_outputs = []

        for game, outcome in data:
            for position, distribution in game:
                legal_moves = self.GameClass.get_legal_moves(position)
                policy = np.zeros_like(legal_moves, dtype=float)
                policy[legal_moves] = distribution

                # convert policy to one-hot
                idx = np.unravel_index(policy.argmax(), policy.shape)
                policy = np.zeros_like(policy)
                policy[idx] = 1

                states.append(position)
                # policy outputs are flattened to match policy output's flat dense output layer
                policy_outputs.append(policy.flatten())
                value_outputs.append(outcome)

        input_data = np.stack(states, axis=0)[:, :, :, :, np.newaxis]
        policy_outputs = np.stack(policy_outputs, axis=0)
        value_outputs = np.array(value_outputs)

        shuffle = np.arange(input_data.shape[0])
        np.random.shuffle(shuffle)
        input_data = input_data[shuffle, ...]
        policy_outputs = policy_outputs[shuffle, ...]
        value_outputs = value_outputs[shuffle, ...]

        return input_data, policy_outputs, value_outputs

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

        while True:
            for policy_pipe in policy_pipes:
                if policy_pipe.poll():
                    policy_pipe.send(network.policy(policy_pipe.recv()))
            for evaluation_pipe in evaluation_pipes:
                if evaluation_pipe.poll():
                    evaluation_pipe.send(network.evaluation(evaluation_pipe.recv()))


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
    net = Network(Connect4)
    net.initialize()
    print('Policy network size: ', net.policy_model.count_params())
    print('Evaluation network size: ', net.evaluation_model.count_params())

    data = []
    for file in os.listdir('../heuristics/games/raw_mcts_games'):
        with open(f'../heuristics/games/raw_mcts_games/{file}', 'rb') as fin:
            data.append(pickle.load(fin))
    net.train(data)
    net.save('../heuristics/models/policy_test.h5', '../heuristics/models/evaluation_test.h5')


if __name__ == '__main__':
    train()
