from .game import Game
import numpy as np
from src.utils.utils import iter_product


class TicTacToe(Game):
    W = 3
    STARTING_STATE = np.stack([np.zeros((W, W)), np.zeros((W, W)), np.ones((W, W))], axis=-1).astype(np.uint8)
    STATE_SHAPE = STARTING_STATE.shape  # 3, 3, 3
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 3, 3, 3
    BOARD_SHAPE = (ROWS, COLUMNS)  # 3, 3
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
        for i, j in iter_product(TicTacToe.BOARD_SHAPE):
            if np.all(state[i, j, :2] == 0):
                move = cls.null_move(state)
                move[i, j, :2] = [1, 0] if cls.is_player_1_turn(state) else [0, 1]
                moves.append(move)
        return moves

    @classmethod
    def get_legal_moves(cls, state):
        return np.array([[np.all(state[i, j, :2] == 0) for j in range(cls.COLUMNS)] for i in range(cls.ROWS)])

    @classmethod
    def is_over(cls, state):
        return cls.check_win(state[:, :, 0]) or cls.check_win(state[:, :, 1]) or cls.is_board_full(state)

    @classmethod
    def get_winner(cls, state):
        if cls.check_win(state[:, :, 0]):
            return 1
        if cls.check_win(state[:, :, 1]):
            return -1
        if cls.is_board_full(state):
            return 0

    @staticmethod
    def check_win(pieces):
        # Check vertical and horizontal
        for k in range(TicTacToe.ROWS):
            if np.all(pieces[k, :]) or np.all(pieces[:, k]):
                return True

        # Check diagonals
        flipped_pieces = np.fliplr(pieces)
        return np.all(np.diag(pieces)) or np.all(np.diag(flipped_pieces))

    @classmethod
    def get_ruleset(cls):
        return f'{TicTacToe.W}x{TicTacToe.W}'
