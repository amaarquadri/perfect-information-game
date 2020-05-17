from src.games.Game import Game
import numpy as np
from src.utils.Utils import iter_product


class Connect4(Game):
    STARTING_STATE = np.stack([np.zeros((6, 7)), np.zeros((6, 7)), np.ones((6, 7))], axis=-1)
    STATE_SHAPE = STARTING_STATE.shape  # 6, 7, 3
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 6, 7
    ROWS, COLUMNS = BOARD_SHAPE
    FEATURE_COUNT = STATE_SHAPE[-1]  # 3
    REPRESENTATION_LETTERS = ['y', 'r']

    def __init__(self, state=STARTING_STATE):
        super().__init__()
        self.state = np.copy(state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = np.copy(state)

    def draw(self, canvas=None):
        if canvas is None:
            print(Connect4.get_human_readable_representation(self.state))
        else:
            raise NotImplementedError()

    @classmethod
    def get_representation_shape(cls):
        return cls.STATE_SHAPE

    @classmethod
    def get_human_readable_representation(cls, state):
        representation = np.full(cls.BOARD_SHAPE, ' ', dtype=str)
        for i in range(cls.FEATURE_COUNT - 1):  # -1 to exclude the turn information
            representation[state[:, :, i] == 1] = cls.REPRESENTATION_LETTERS[i]
        return representation

    @classmethod
    def get_starting_state(cls):
        return np.copy(cls.STARTING_STATE)

    @classmethod
    def get_possible_moves(cls, state):
        moves = []
        combined_board = np.logical_or(state[:, :, 0], state[:, :, 1])
        for j in range(cls.COLUMNS):
            if np.all(state[0, j, :2] == 0):
                column = combined_board[:, j]
                max_empty_i = cls.ROWS - 1 if np.all(column == 0) else np.argmax(np.diff(column))

                move = cls.null_move(state)
                move[max_empty_i, j, :2] = [1, 0] if cls.is_player_1_turn(state) else [0, 1]
                moves.append(move)
        return moves

    @classmethod
    def is_player_1_turn(cls, state):
        return np.all(state[:, :, -1])

    @classmethod
    def check_win(cls, pieces):
        # Check vertical
        for i, j in iter_product((cls.ROWS, cls.COLUMNS - 4)):
            if np.all(pieces[i, j:j + 4]):
                return True

        # Check horizontal
        for i, j in iter_product((cls.ROWS - 4, cls.COLUMNS)):
            if np.all(pieces[i:i + 4, j]):
                return True

        # Check diagonals
        flipped_pieces = np.fliplr(pieces)
        for i, j in iter_product((cls.ROWS - 4, cls.COLUMNS - 4)):
            if np.all(np.diag(pieces[i:i + 4, j:j + 4])) or np.all(np.diag(flipped_pieces[i:i + 4, j:j + 4])):
                return True

    @classmethod
    def full_board(cls, state):
        return np.all(np.logical_or(state[:, :, 0], state[:, :, 1]) == 1)

    @classmethod
    def is_over(cls, state):
        return cls.check_win(state[:, :, 0]) or cls.check_win(state[:, :, 1]) or cls.full_board(state)

    @classmethod
    def get_winner(cls, state):
        if cls.check_win(state[:, :, 0]):
            return 1
        if cls.check_win(state[:, :, 1]):
            return -1
        if cls.full_board(state):
            return 0

    @classmethod
    def null_move(cls, state):
        move = np.copy(state)
        move[:, :, -1] = np.zeros(cls.BOARD_SHAPE) if cls.is_player_1_turn(state) \
            else np.ones(cls.BOARD_SHAPE)
        return move
