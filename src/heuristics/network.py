from time import time
import numpy as np
from multiprocessing import Process, Pipe
import sys
from signal import signal, SIGTERM
from src.utils.utils import get_training_path


class Network:
    """
    A deep convolutional neural network that takes as input the ML representation of a game (n, m, k) and tries to
    return 1 if player 1 (white in chess, red in checkers etc.) is going to win, -1 if player 2 is going to win,
    and 0 if the game is going to be a draw.

    Network will receive all inputs and output moves in the format of flattened arrays of legal moves. \
    It will internally handle conversion too and from move distribution matrices using GameClass.get_legal_moves
    """

    def __init__(self, GameClass, model_path=None, reinforcement_training=False):
        self.GameClass = GameClass
        self.model_path = model_path

        # lazily initialized so Network can be passed between processes before being initialized
        self.model = None

        self.reinforcement_training = reinforcement_training
        self.tensor_board = None
        self.epoch = 0

    def initialize(self):
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.models import load_model
        from keras.callbacks import TensorBoard

        if self.model is not None:
            return

        if self.model_path is not None:
            input_shape = self.GameClass.STATE_SHAPE
            output_shape = self.GameClass.MOVE_SHAPE
            self.model = load_model(self.model_path)
            if self.model.input_shape != (None,) + input_shape:
                raise Exception('Input shape of loaded model doesn\'t match!')
            if self.model.output_shape != [(None,) + output_shape, (None, 1)]:
                raise Exception('Output shape of loaded model doesn\'t match!')
        else:
            self.model = self.create_model()

        if self.reinforcement_training:
            self.tensor_board = TensorBoard(log_dir=f'{get_training_path(self.GameClass)}/logs/'
                                                    f'model_reinforcement_{time()}',
                                            histogram_freq=0, batch_size=256, write_graph=True, write_grads=True)
            self.tensor_board.set_model(self.model)

    def create_model(self, kernel_size=(4, 4), residual_layers=6):
        """
        https://www.youtube.com/watch?v=OPgRNY3FaxA
        """
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.models import Model
        from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Activation, Add, Reshape

        input_shape = self.GameClass.STATE_SHAPE
        output_shape = self.GameClass.MOVE_SHAPE
        output_neurons = np.product(output_shape)

        input_tensor = Input(input_shape)

        # convolutional layer
        x = Conv2D(16, kernel_size, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # residual layers
        for _ in range(residual_layers):
            y = Conv2D(16, kernel_size, padding='same')(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(16, kernel_size, padding='same')(y)
            y = BatchNormalization()(y)
            # noinspection PyTypeChecker
            x = Add()([x, y])
            x = Activation('relu')(x)

        # policy head
        policy = Conv2D(2, (1, 1), padding='same')(x)
        policy = BatchNormalization()(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(output_neurons, activation='softmax')(policy)
        policy = Reshape(output_shape, name='policy')(policy)

        # value head
        value = Conv2D(1, (1, 1), padding='same')(x)
        value = BatchNormalization()(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(16, activation='relu')(value)
        value = Dense(1, activation='tanh', name='value')(value)

        model = Model(input_tensor, [policy, value])
        model.compile(optimizer='adam', loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                      metrics=['mean_squared_error'])
        return model

    def predict(self, states):
        return self.model.predict(states)

    def call(self, states):
        """
        For any of the given states, if no moves are legal, then the corresponding probability distribution will be a
        list with a single 1. This is done to allow for pass moves which are not encapsulated by GameClass.MOVE_SHAPE.

        :param states: The input positions with shape (k,) + GameClass.State_Shape, where k is the number of positions.
        :return: A list of length k. Each element of the list is a tuple where the 0th element is the probability
                 distribution on legal moves, and the 1st element is the evaluation (a float in (-1, 1)).
        """
        raw_policies, evaluations = self.predict(states)

        filtered_policies = [raw_policy[self.GameClass.get_legal_moves(state)]
                             for state, raw_policy in zip(states, raw_policies)]
        filtered_policies = [filtered_policy / np.sum(filtered_policy) if len(filtered_policy) > 0 else [1]
                             for filtered_policy in filtered_policies]

        evaluations = evaluations.reshape(states.shape[0])
        return [(filtered_policy, evaluation) for filtered_policy, evaluation in zip(filtered_policies, evaluations)]

    def choose_move(self, position, return_evaluation=False, optimal=False):
        distribution, evaluation = self.call(position[np.newaxis, ...])[0]
        idx = np.argmin(distribution) if optimal else np.random.choice(np.arange(len(distribution)), p=distribution)
        move = self.GameClass.get_possible_moves(position)[idx]
        return (move, evaluation) if return_evaluation else move

    def train(self, data, validation_fraction=0.2):
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.callbacks import TensorBoard, EarlyStopping

        split = int((1 - validation_fraction) * len(data))
        train_input, train_output = self.process_data(self.GameClass, data[:split])
        test_input, test_output = self.process_data(self.GameClass, data[split:])
        print('Training Samples:', train_input.shape[0])
        print('Validation Samples:', test_input.shape[0])

        self.model.fit(train_input, train_output, epochs=100, validation_data=(test_input, test_output),
                       callbacks=[TensorBoard(log_dir=f'{get_training_path(self.GameClass)}/logs/model_{time()}'),
                                  EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

    def train_step(self, states, policies, values):
        logs = self.model.train_on_batch(states, [policies, values], return_dict=True)
        self.tensor_board.on_epoch_end(self.epoch, logs)
        self.epoch += 1

    def finish_training(self):
        self.tensor_board.on_train_end()

    def save(self, model_path):
        self.model.save(model_path)

    def equal_model_architecture(self, network):
        """
        Both networks must be initialized.

        :return: True if this Network's model and the given network's model have the same architecture.
        """
        return self.model.get_config() == network.model.get_config()

    @classmethod
    def process_data(cls, GameClass, data, one_hot=False, shuffle=True):
        states = []
        policy_outputs = []
        value_outputs = []

        for game, outcome in data:
            for position, distribution in game:
                legal_moves = GameClass.get_legal_moves(position)
                policy = np.zeros_like(legal_moves, dtype=float)
                policy[legal_moves] = distribution

                if one_hot:
                    idx = np.unravel_index(policy.argmax(), policy.shape)
                    policy = np.zeros_like(policy)
                    policy[idx] = 1

                states.append(position)
                policy_outputs.append(policy)
                value_outputs.append(outcome)

        input_data = np.stack(states, axis=0)
        policy_outputs = np.stack(policy_outputs, axis=0)
        value_outputs = np.array(value_outputs)

        if shuffle:
            shuffle_indices = np.arange(input_data.shape[0])
            np.random.shuffle(shuffle_indices)
            input_data = input_data[shuffle_indices, ...]
            policy_outputs = policy_outputs[shuffle_indices, ...]
            value_outputs = value_outputs[shuffle_indices]

        return input_data, [policy_outputs, value_outputs]

    @staticmethod
    def spawn_dual_architecture_process(GameClass, model_path=None, pipes_per_section=1):
        parent_a_pipes, worker_a_pipes = zip(*[Pipe() for _ in range(pipes_per_section)])
        parent_b_pipes, worker_b_pipes = zip(*[Pipe() for _ in range(pipes_per_section)])
        worker_training_data_pipe, parent_training_data_pipe = Pipe(duplex=False)
        process = Process(target=Network.dual_architecture_process_loop,
                          args=(GameClass, model_path, worker_a_pipes, worker_b_pipes, worker_training_data_pipe))
        proxy_a_networks = [ProxyNetwork(GameClass, parent_a_pipe) for parent_a_pipe in parent_a_pipes]
        proxy_b_networks = [ProxyNetwork(GameClass, parent_b_pipe) for parent_b_pipe in parent_b_pipes]
        return process, proxy_a_networks, proxy_b_networks, parent_training_data_pipe

    @staticmethod
    def dual_architecture_process_loop(GameClass, model_path, worker_a_pipes, worker_b_pipes, training_data_pipe):
        network = Network(GameClass, model_path, reinforcement_training=True)
        network.initialize()

        def on_terminate_process(_):
            network.finish_training()
            network.save(model_path)
            sys.exit(0)
        signal(SIGTERM, on_terminate_process)

        def send_results(requests, results, pipes):
            raw_policies, evaluations = results
            pos = 0
            for request, pipe in zip(requests, pipes):
                new_pos = pos + request.shape[0]
                # This send call will not block because the receiver will always be waiting to read the result
                pipe.send((raw_policies[pos:new_pos], evaluations[pos:new_pos]))
                pos = new_pos

        requests_a = [worker_a_pipe.recv() for worker_a_pipe in worker_a_pipes]
        # concatenate is used instead of stack because each request already has shape (k,) + GameClass.STATE_SHAPE
        results_a = network.predict(np.concatenate(requests_a, axis=0))

        last_save = time()
        while True:
            if training_data_pipe.poll():
                data = training_data_pipe.recv()
                states, policies, values = data
                network.train_step(states, policies, values)
                if time() - last_save > 5 * 60:
                    network.save(model_path)
                    last_save = time()

            # receive B requests
            requests_b = [worker_b_pipe.recv() for worker_b_pipe in worker_b_pipes]

            # return A results
            send_results(requests_a, results_a, worker_a_pipes)

            # compute B results
            results_b = network.predict(np.concatenate(requests_b, axis=0))

            # receive A requests
            requests_a = [worker_a_pipe.recv() for worker_a_pipe in worker_a_pipes]

            # return B results
            send_results(requests_b, results_b, worker_b_pipes)

            # compute A results
            results_a = network.predict(np.concatenate(requests_a, axis=0))


class ProxyNetwork(Network):
    def __init__(self, GameClass, model_pipe):
        super().__init__(GameClass)
        self.model_pipe = model_pipe

    def initialize(self):
        pass

    def predict(self, states):
        self.model_pipe.send(states)
        return self.model_pipe.recv()

    def create_model(self, kernel_size=(4, 4), residual_layers=6):
        raise NotImplementedError('ProxyNetwork does not support this operation!')

    def train(self, data, validation_fraction=0.2):
        raise NotImplementedError('ProxyNetwork does not support this operation!')

    def save(self, model_path):
        raise NotImplementedError('ProxyNetwork does not support this operation!')

    def equal_model_architecture(self, network):
        raise NotImplementedError('ProxyNetwork does not support this operation!')


def train_from_scratch():
    from src.utils.active_game import ActiveGame as GameClass
    import os
    import pickle
    net = Network(GameClass)
    net.initialize()
    print('Network size: ', net.model.count_params())

    data = []
    for file in sorted(os.listdir(f'{get_training_path(GameClass)}/games/rollout_mcts_games')):
        if file[-7:] != '.pickle':
            continue
        with open(f'{get_training_path(GameClass)}/games/rollout_mcts_games/{file}', 'rb') as fin:
            data.append(pickle.load(fin))
    net.train(data)

    data = []
    for file in sorted(os.listdir(f'{get_training_path(GameClass)}/games/reinforcement_learning_games')):
        if file[-7:] != '.pickle':
            continue
        with open(f'{get_training_path(GameClass)}/games/reinforcement_learning_games/{file}', 'rb') as fin:
            data.append(pickle.load(fin))

    sets = int(len(data) / 1200)
    for k in range(sets):
        net.train(data[k * len(data) // sets:(k + 1) * len(data) // sets])

    net.save(f'{get_training_path(GameClass)}/models/model_reinforcement.h5')


if __name__ == '__main__':
    train_from_scratch()
