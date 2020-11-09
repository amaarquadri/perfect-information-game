from .game import Game
import numpy as np
from ..utils.utils import iter_product


class Connect4(Game):
    STARTING_STATE = np.stack([np.zeros((6, 7)), np.zeros((6, 7)), np.ones((6, 7))], axis=-1).astype(np.uint8)
    STATE_SHAPE = STARTING_STATE.shape  # 6, 7, 3
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 6, 7, 3
    BOARD_SHAPE = (ROWS, COLUMNS)  # 6, 7
    MOVE_SHAPE = (COLUMNS,)
    REPRESENTATION_LETTERS = ['y', 'r']
    REPRESENTATION_FILES = ['dark_square', 'yellow_circle_dark_square', 'red_circle_dark_square']
    CLICKS_PER_MOVE = 1

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        j = clicks[0][1]
        combined_column = np.logical_or(self.state[:, j, 0], self.state[:, j, 1])
        max_empty_i = Connect4.ROWS - 1 if np.all(combined_column == 0) else np.argmax(np.diff(combined_column))
        new_state = self.null_move(self.state)
        new_state[max_empty_i, j, :2] = [1, 0] if self.is_player_1_turn(self.state) else [0, 1]

        for legal_move in Connect4.get_possible_moves(self.state):
            if np.all(legal_move == new_state):
                break
        else:
            raise ValueError('Illegal Move')

        self.state = new_state

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
    def get_legal_moves(cls, state):
        return np.array([np.all(state[0, j, :2] == 0) for j in range(cls.COLUMNS)])

    @classmethod
    def check_win(cls, pieces):
        # Check vertical
        for i, j in iter_product((cls.ROWS, cls.COLUMNS - 3)):
            if np.all(pieces[i, j:j + 4]):
                return True

        # Check horizontal
        for i, j in iter_product((cls.ROWS - 3, cls.COLUMNS)):
            if np.all(pieces[i:i + 4, j]):
                return True

        # Check diagonals
        flipped_pieces = np.fliplr(pieces)
        for i, j in iter_product((cls.ROWS - 3, cls.COLUMNS - 3)):
            if np.all(np.diag(pieces[i:i + 4, j:j + 4])) or np.all(np.diag(flipped_pieces[i:i + 4, j:j + 4])):
                return True

        return False

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
