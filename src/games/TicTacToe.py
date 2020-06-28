from src.games.Game import Game
import numpy as np
from src.utils.Utils import iter_product


class TicTacToe(Game):
    STARTING_STATE = np.stack([np.zeros((3, 3)), np.zeros((3, 3)), np.ones((3, 3))], axis=-1)
    STATE_SHAPE = STARTING_STATE.shape  # 3, 3, 3
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 3, 3
    ROWS, COLUMNS = BOARD_SHAPE
    BOARD_LENGTH = BOARD_SHAPE[0]  # 3
    FEATURE_COUNT = STATE_SHAPE[-1]  # 3
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
        for k in range(3):
            if np.all(pieces[k, :]) or np.all(pieces[:, k]):
                return True

        # Check diagonals
        flipped_pieces = np.fliplr(pieces)
        return np.all(np.diag(pieces)) or np.all(np.diag(flipped_pieces))
