from itertools import product
from src.games.Game import Game
import numpy as np
from src.utils.Utils import iter_product


class Checkers(Game):
    RED = np.array(5 * [8 * [0]] + [4 * [1, 0]] + [4 * [0, 1]] + [4 * [1, 0]])
    BLACK = np.array([4 * [0, 1]] + [4 * [1, 0]] + [4 * [0, 1]] + 5 * [8 * [0]])
    STARTING_STATE = np.stack([RED, np.zeros((8, 8)), BLACK, np.zeros((8, 8)), np.ones((8, 8))], axis=-1)
    STATE_SHAPE = STARTING_STATE.shape  # 8, 8, 5
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 8, 8
    BOARD_LENGTH = BOARD_SHAPE[0]  # 8
    FEATURE_COUNT = STATE_SHAPE[-1]  # 5
    REPRESENTATION_LETTERS = ['r', 'R', 'b', 'B']

    def __init__(self, state=STARTING_STATE):
        super().__init__()
        self.state = np.copy(state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = np.copy(state)

    def draw(self, canvas=None):
        if canvas is None:
            print(Checkers.get_human_readable_representation(self.state))
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
        return cls.STARTING_STATE

    @classmethod
    def get_possible_moves(cls, state):
        """
        Moveset: kings can only move 1 square. Multi jumps are not allowed.
        :return:
        """
        # TODO: include multi captures
        moves = []
        if cls.is_player_1_turn(state):
            for i, j, (di, dj, king_move) in iter_product(cls.BOARD_SHAPE, [(-1, -1, False), (-1, 1, False),
                                                                                 (1, -1, True), (1, 1, True)]):
                if np.any(state[i, j, 1] if king_move else state[i, j, :2]) \
                        and cls.is_valid(i + di, j + dj):
                    if cls.is_empty(state, i + di, j + dj):
                        move = cls.null_move(state)
                        move[i + di, j + dj, :2] = move[i, j, :2] if i + di > 0 else [0, 1]
                        move[i, j, :2] = [0, 0]
                        moves.append(move)
                    elif cls.is_valid(i + 2 * di, j + 2 * dj) \
                            and cls.is_empty(state, i + 2 * di, j + 2 * dj) \
                            and np.any(state[i + di, j + dj, 2:4]):
                        move = cls.null_move(state)
                        move[i + 2 * di, j + 2 * dj, :2] = move[i, j, :2] if i + 2 * di > 0 else [0, 1]
                        move[i, j, :2] = [0, 0]
                        move[i + di, j + dj, 2:4] = [0, 0]
                        moves.append(move)
        else:
            for i, j, (di, dj, king_move) in iter_product(cls.BOARD_SHAPE, [(1, -1, False), (1, 1, False),
                                                                                 (-1, -1, True), (-1, 1, True)]):
                if np.any(state[i, j, 3] if king_move else state[i, j, 2:4]) \
                        and cls.is_valid(i + di, j + dj):
                    if cls.is_empty(state, i + di, j + dj):
                        move = cls.null_move(state)
                        move[i + di, j + dj, 2:4] = move[i, j, 2:4] if i + di < 7 else [0, 1]
                        move[i, j, 2:4] = [0, 0]
                        moves.append(move)
                    elif cls.is_valid(i + 2 * di, j + 2 * dj) \
                            and cls.is_empty(state, i + 2 * di, j + 2 * dj) \
                            and np.any(state[i + di, j + dj, :2]):
                        move = cls.null_move(state)
                        move[i + 2 * di, j + 2 * dj, 2:4] = move[i, j, 2:4] if i + 2 * di < 7 else [0, 1]
                        move[i, j, 2:4] = [0, 0]
                        move[i + di, j + dj, :2] = [0, 0]
                        moves.append(move)
        return moves

    @classmethod
    def is_over(cls, state):
        return len(cls.get_possible_moves(state)) == 0

    @classmethod
    def get_winner(cls, state):
        if not cls.is_over(state):
            raise Exception('Game is not over!')
        if cls.is_player_1_turn(state):
            return -1
        else:
            return 1

    @classmethod
    def is_valid(cls, i, j):
        return 0 <= i < cls.BOARD_LENGTH and 0 <= j < cls.BOARD_LENGTH

    @classmethod
    def is_empty(cls, state, i, j):
        return np.all(state[i, j, :-1] == 0)
