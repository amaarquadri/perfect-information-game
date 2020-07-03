from src.games.Game import Game
import numpy as np
from src.utils.Utils import iter_product


class Checkers(Game):
    RED = np.array(5 * [8 * [0]] + [4 * [1, 0]] + [4 * [0, 1]] + [4 * [1, 0]])
    BLACK = np.array([4 * [0, 1]] + [4 * [1, 0]] + [4 * [0, 1]] + 5 * [8 * [0]])
    STARTING_STATE = np.stack([RED, np.zeros((8, 8)),
                               BLACK, np.zeros((8, 8)),
                               np.ones((8, 8))], axis=-1).astype(np.uint8)
    STATE_SHAPE = STARTING_STATE.shape  # 8, 8, 5
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 8, 8
    ROWS, COLUMNS = BOARD_SHAPE
    BOARD_LENGTH = BOARD_SHAPE[0]  # 8
    FEATURE_COUNT = STATE_SHAPE[-1]  # 5
    MOVE_SHAPE = (ROWS // 2, COLUMNS // 2, 4)
    REPRESENTATION_LETTERS = ['r', 'R', 'b', 'B']
    CLICKS_PER_MOVE = 2
    REPRESENTATION_FILES = ['dark_square', 'red_circle_dark_square', 'red_circle_k_dark_square',
                            'black_circle_dark_square', 'black_circle_k_dark_square']

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        (start_i, start_j), (end_i, end_j) = clicks
        new_state = self.null_move(self.state)

        is_king = np.any(self.state[start_i, start_j, [1, 3]] == 1) or \
            end_j == 0 or end_j == Checkers.ROWS - 1
        new_piece = [0] * 4
        new_piece[2 * (not self.is_player_1_turn(self.state)) + is_king] = 1
        new_state[end_i, end_j, :4] = new_piece
        new_state[start_i, start_j, :4] = [0] * 4
        if np.abs(end_j - start_j) == 2:
            new_state[(start_i + end_i) // 2, (start_j + end_j) // 2, :4] = [0] * 4

        for move in self.get_possible_moves(self.state):
            if np.all(move == new_state):
                break
        else:
            raise ValueError('Invalid Move!')

        self.state = new_state

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
    def get_legal_moves(cls, state):
        raise NotImplementedError()

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
