from src.games.game import Game
import numpy as np
from src.utils.utils import iter_product


class MultiTicTacToe(Game):
    # TODO: finish coding, and debugging
    D = 3
    W = 3
    STARTING_STATE = np.stack([np.zeros(D * [W]), np.zeros(D * [W]), np.ones(D * [W])], axis=-1).astype(np.uint8)
    STATE_SHAPE = STARTING_STATE.shape  # 3, 3, 3
    FEATURE_COUNT = STATE_SHAPE[-1]  # 3, 3, 3
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 3, 3
    MOVE_SHAPE = BOARD_SHAPE
    REPRESENTATION_LETTERS = ['X', 'O']
    CLICKS_PER_MOVE = 1
    REPRESENTATION_FILES = ['dark_square', 'white_circle_dark_square', 'black_circle_dark_square']

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        i, j = clicks[0]
        if np.any(self.state[i, j, :2] != 0):
            raise ValueError('Invalid Move!')

        new_state = self.null_move(self.state)
        new_state[i, j, :2] = [1, 0] if self.is_player_1_turn(self.state) else [0, 1]
        self.state = new_state

    @classmethod
    def get_possible_moves(cls, state):
        moves = []
        for coords in iter_product(MultiTicTacToe.BOARD_SHAPE):
            if state[coords + (0,)] == 0 and state[coords + (1,)] == 0:
                move = cls.null_move(state)
                if cls.is_player_1_turn(state):
                    move[coords + (0,)] = 1
                else:
                    move[coords + (1,)] = 1
                moves.append(move)
        return moves

    @classmethod
    def get_legal_moves(cls, state):
        return np.array([state[coords + (0,)] == 0 and state[coords + (1,)] == 0
                         for coords in iter_product(MultiTicTacToe.BOARD_SHAPE)]) \
            .reshape(MultiTicTacToe.BOARD_SHAPE)
        # return np.array([[np.all(state[i, j, :2] == 0) for coords in iter_product(MultiTicTacToe.BOARD_SHAPE))

    @classmethod
    def is_over(cls, state):
        return cls.check_win(state[..., 0]) or cls.check_win(state[..., 1]) or cls.is_board_full(state)

    @classmethod
    def get_winner(cls, state):
        if cls.check_win(state[..., 0]):
            return 1
        if cls.check_win(state[..., 1]):
            return -1
        if cls.is_board_full(state):
            return 0

    @staticmethod
    def check_win(pieces):
        for indices in MultiTicTacToe.generator():
            for index in indices:
                if pieces[tuple(index)] == 0:
                    break
            else:
                return True
        return False

    @staticmethod
    def generator():
        # This function will double count:
        # i.e. each valid winning line will be returned twice: once forwards and once backwards

        increasing = np.arange(MultiTicTacToe.W)
        decreasing = increasing[::-1]
        for coord_states in iter_product(MultiTicTacToe.D * [MultiTicTacToe.W + 2]):
            # coords_states is a tuple with length D, each element indicates whether
            # that dimension is a specific constant value, increasing, or decreasing
            indices = np.zeros((MultiTicTacToe.W, MultiTicTacToe.D), dtype=int)
            stationary = True
            for i, coord_state in enumerate(coord_states):
                if coord_state < MultiTicTacToe.W:
                    vals = np.full((MultiTicTacToe.D,), coord_state)
                elif coord_state == MultiTicTacToe.W:
                    vals = increasing
                    stationary = False
                else:
                    vals = decreasing
                    stationary = False
                indices[:, i] = vals

            # if none of the indices are increasing or decreasing, then all spots correspond to the exact same square
            if not stationary:
                yield indices
