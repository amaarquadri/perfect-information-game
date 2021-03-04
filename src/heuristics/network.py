from time import time
import numpy as np
from utils.utils import get_training_path


class Network:
    """
    A deep convolutional neural network that takes as input the ML representation of a game (n, m, k) and tries to
    return 1 if player 1 (white in chess, red in checkers etc.) is going to win, -1 if player 2 is going to win,
    and 0 if the game is going to be a draw.

    The Network's model will input a binary matrix with a shape of GameClass.STATE_SHAPE and output a tuple consisting
    of the probability distribution over legal moves and the position's evaluation.
    """

    def __init__(self, GameClass, model_path=None, reinforcement_training=False, hyper_params=None):
        self.GameClass = GameClass
        self.model_path = model_path

        # lazily initialized so Network can be passed between processes before being initialized
        self.model = None

        self.reinforcement_training = reinforcement_training
        self.hyper_params = hyper_params if hyper_params is not None else {}
        self.tensor_board = None
        self.epoch = 0

    def initialize(self):
        """
        Initializes the Network's model.
        """
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
            # TODO: recompile model with loss_weights and learning schedule from config file
        else:
            self.model = self.create_model(**self.hyper_params)

        if self.reinforcement_training:
            self.tensor_board = TensorBoard(log_dir=f'{get_training_path(self.GameClass)}/logs/'
                                                    f'model_reinforcement_{time()}',
                                            histogram_freq=0, write_graph=True)
            self.tensor_board.set_model(self.model)

    def create_model(self, kernel_size=(4, 4), convolutional_filters=64, residual_layers=6,
                     value_head_neurons=16, policy_loss_value=1):
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
        x = Conv2D(convolutional_filters, kernel_size, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # residual layers
        for _ in range(residual_layers):
            y = Conv2D(convolutional_filters, kernel_size, padding='same')(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(convolutional_filters, kernel_size, padding='same')(y)
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
        value = Dense(value_head_neurons, activation='relu')(value)
        value = Dense(1, activation='tanh', name='value')(value)

        model = Model(input_tensor, [policy, value])
        model.compile(optimizer='adam', loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
                      loss_weights={'policy': policy_loss_value, 'value': 1},
                      metrics=['mean_squared_error'])
        return model

    def predict(self, states):
        return self.model.predict(states)

    def call(self, states):
        """
        For any of the given states, if no moves are legal, then the corresponding probability distribution will be a
        list with a single 1. This is done to allow for pass moves which are not encapsulated by GameClass.MOVE_SHAPE.

        :param states: The input positions with shape (k,) + GameClass.STATE_SHAPE, where k is the number of positions.
        :return: A list of length k. Each element of the list is a tuple where the 0th element is the probability
                 distribution on legal moves (ordered correspondingly with GameClass.get_possible_moves), and the 1st
                 element is the evaluation (a float in (-1, 1)).
        """
        raw_policies, evaluations = self.predict(states)

        filtered_policies = [raw_policy[self.GameClass.get_legal_moves(state)]
                             for state, raw_policy in zip(states, raw_policies)]
        filtered_policies = [filtered_policy / np.sum(filtered_policy) if len(filtered_policy) > 0 else [1]
                             for filtered_policy in filtered_policies]

        evaluations = evaluations.reshape(states.shape[0])
        return [(filtered_policy, evaluation) for filtered_policy, evaluation in zip(filtered_policies, evaluations)]

    def choose_move(self, position, return_distribution=False, optimal=False):
        distribution, evaluation = self.call(position[np.newaxis, ...])[0]
        idx = np.argmin(distribution) if optimal else np.random.choice(np.arange(len(distribution)), p=distribution)
        move = self.GameClass.get_possible_moves(position)[idx]
        return (move, distribution) if return_distribution else move

    def train(self, data, validation_fraction=0.2):
        # Note: keras imports are within functions to prevent initializing keras in processes that import from this file
        from keras.callbacks import TensorBoard, EarlyStopping

        split = int((1 - validation_fraction) * len(data))
        train_input, train_output = self.get_training_data(self.GameClass, data[:split])
        test_input, test_output = self.get_training_data(self.GameClass, data[split:])
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
    def get_training_data(cls, GameClass, data, one_hot=False, shuffle=True):
        """


        :param GameClass:
        :param data: A list of game, outcome tuples. Each game is a list of position, distribution tuples.
        :param one_hot:
        :param shuffle:
        :return:
        """
        states = []
        policy_outputs = []
        value_outputs = []

        for game, outcome in data:
            for position, distribution in game:
                legal_moves = GameClass.get_legal_moves(position)
                policy = np.zeros_like(legal_moves, dtype=float)
                policy[legal_moves] = distribution
                policy /= np.sum(policy)  # rescale so total probability is 1

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

