from src.games.Game import Game
import numpy as np
from src.utils.Utils import iter_product


class TicTacToe(Game):
    STARTING_STATE = np.stack([np.zeros((3, 3)), np.zeros((3, 3)), np.ones((3, 3))], axis=-1)
    STATE_SHAPE = STARTING_STATE.shape  # 3, 3, 3
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 3, 3
    BOARD_LENGTH = BOARD_SHAPE[0]  # 3
    FEATURE_COUNT = STATE_SHAPE[-1]  # 3
    REPRESENTATION_LETTERS = ['X', 'O']

    def __init__(self, state=STARTING_STATE):
        super().__init__()
        self.state = state

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def perform_user_move(self, move):
        move_number = int(move)
        empty_square_index = 1
        for i, j in iter_product(TicTacToe.BOARD_SHAPE):
            if np.all(self.state[i, j, :-1] == 0):
                if empty_square_index == move_number:
                    new_state = self.null_move(self.state)
                    new_state[i, j, :2] = [1, 0] if self.is_player_1_turn(self.state) else [0, 1]
                    self.state = new_state
                    return
                else:
                    empty_square_index += 1
        raise ValueError('No move matching specified string')

    def draw(self, canvas=None, move_prompt=False):
        if canvas is None:
            print(TicTacToe.get_human_move_prompt_representation(self.state) if move_prompt else
                  TicTacToe.get_human_readable_representation(self.state))
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
    def get_human_move_prompt_representation(cls, state):
        representation = cls.get_human_readable_representation(state)
        move_number = 1
        for i, j in iter_product(TicTacToe.BOARD_SHAPE):
            if np.all(state[i, j, :-1] == 0):
                representation[i, j] = str(move_number)
                move_number += 1
        return representation

    @classmethod
    def get_starting_state(cls):
        return cls.STARTING_STATE

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
        return cls.check_win(state[:, :, 0]) or cls.check_win(state[:, :, 1]) or cls.full_board(state)

    @classmethod
    def get_winner(cls, state):
        if cls.check_win(state[:, :, 0]):
            return 1
        if cls.check_win(state[:, :, 1]):
            return -1
        if cls.full_board(state):
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

    @staticmethod
    def full_board(state):
        return np.all(np.logical_or(state[:, :, 0], state[:, :, 1]) == 1)
