from src.games.game import Game
import numpy as np
from src.utils.utils import iter_product


class Amazons(Game):
    WHITE_STARTING_BOARD_6x6 = np.array([[0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0]])
    WHITE_STARTING_BOARD_4x4 = np.array([[0, 0, 1, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 1, 0, 0]])
    WHITE_STARTING_BOARD = WHITE_STARTING_BOARD_4x4
    BLACK_STARTING_BOARD = np.rot90(WHITE_STARTING_BOARD)
    STARTING_STATE = np.stack([WHITE_STARTING_BOARD,
                               BLACK_STARTING_BOARD,
                               np.zeros_like(WHITE_STARTING_BOARD),
                               np.ones_like(WHITE_STARTING_BOARD)], axis=-1).astype(np.uint8)

    STATE_SHAPE = STARTING_STATE.shape  # 6, 6, 4
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 6, 6, 3
    BOARD_SHAPE = (ROWS, COLUMNS)  # 6, 6
    # TODO: figure out how to deal with very sparse arrays
    MOVE_SHAPE = (ROWS, COLUMNS, 8 * (ROWS - 1), 8 * (ROWS - 1))  # assumes ROWS == COLUMNS
    REPRESENTATION_LETTERS = ['W', 'B', 'X']
    REPRESENTATION_FILES = ['dark_square', 'white_circle_dark_square',
                            'black_circle_dark_square', 'red_circle_dark_square']
    CLICKS_PER_MOVE = 3

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        (piece_from_i, piece_from_j), (piece_to_i, piece_to_j), (shot_i, shot_j) = clicks
        new_state = self.null_move(self.state)
        new_state[piece_to_i, piece_to_j, :2] = new_state[piece_from_i, piece_from_j, :2]
        new_state[piece_from_i, piece_from_j, :2] = [0, 0]
        new_state[shot_i, shot_j, 2] = 1

        for legal_move in Amazons.get_possible_moves(self.state):
            if np.all(legal_move == new_state):
                break
        else:
            raise ValueError('Illegal Move')

        self.state = new_state

    @staticmethod
    def is_valid(i, j):
        return 0 <= i < Amazons.ROWS and 0 <= j < Amazons.COLUMNS

    @staticmethod
    def shoot(state, i, j):
        targets = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            p_x, p_y = i + di, j + dj
            while Amazons.is_valid(p_x, p_y) and np.all(state[p_x, p_y, :3] == 0):
                targets.append((p_x, p_y))
                p_x += di
                p_y += dj
        return targets

    @classmethod
    def get_possible_moves(cls, state):
        moves = []

        player_index = 0 if cls.is_player_1_turn(state) else 1
        for i, j in iter_product(Amazons.BOARD_SHAPE):
            if state[i, j, player_index] == 1:
                for p_x, p_y in Amazons.shoot(state, i, j):
                    partial_move = cls.null_move(state)
                    partial_move[i, j, player_index] = 0
                    partial_move[p_x, p_y, player_index] = 1
                    for t_x, t_y in Amazons.shoot(partial_move, p_x, p_y):
                        full_move = np.copy(partial_move)  # Don't use null_move because turn was already switched above
                        full_move[t_x, t_y, 2] = 1
                        moves.append(full_move)
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
        return -1 if cls.is_player_1_turn(state) else 1
