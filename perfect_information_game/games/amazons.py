from perfect_information_game.games import Game
import numpy as np
from perfect_information_game.utils import iter_product, DIRECTIONS_8


class Amazons(Game):
    CONFIG = '6x6'
    # CONFIG = '4x4'

    PLAYER_1_STARTING_BOARD_6x6 = np.array([[0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0]])
    PLAYER_1_STARTING_BOARD_4x4 = np.array([[0, 0, 1, 0],
                                            [0, 0, 0, 0],
                                            [0, 0, 0, 0],
                                            [0, 1, 0, 0]])
    PLAYER_1_STARTING_BOARD = PLAYER_1_STARTING_BOARD_6x6 if CONFIG == '6x6' else PLAYER_1_STARTING_BOARD_4x4
    PLAYER_2_STARTING_BOARD = np.rot90(PLAYER_1_STARTING_BOARD)
    STARTING_STATE = np.stack([PLAYER_1_STARTING_BOARD,
                               PLAYER_2_STARTING_BOARD,
                               np.zeros_like(PLAYER_1_STARTING_BOARD),
                               np.ones_like(PLAYER_1_STARTING_BOARD)], axis=-1).astype(np.uint8)

    STATE_SHAPE = STARTING_STATE.shape  # 6, 6, 4 or 4, 4, 4
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 6, 6, 3 or 4, 4, 3
    BOARD_SHAPE = (ROWS, COLUMNS)  # 6, 6 or 4, 4
    # TODO: figure out how to deal with very sparse arrays
    # assumes ROWS == COLUMNS
    MOVE_SHAPE = (ROWS, COLUMNS, 8, (ROWS - 1), 8, (ROWS - 1))  # 6, 6, 8, 5, 8, 5 or 4, 4, 8, 3, 8, 3
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

        for legal_move in self.get_possible_moves(self.state):
            if np.all(legal_move == new_state):
                break
        else:
            raise ValueError('Illegal Move')

        self.state = new_state

    @classmethod
    def shoot(cls, state, i, j):
        targets = []
        for di, dj in DIRECTIONS_8:
            p_x, p_y = i + di, j + dj
            while cls.is_valid(p_x, p_y) and np.all(state[p_x, p_y, :3] == 0):
                targets.append((p_x, p_y))
                p_x += di
                p_y += dj
        return targets

    @classmethod
    def get_possible_moves(cls, state):
        moves = []

        player_index = 0 if cls.is_player_1_turn(state) else 1
        for i, j in iter_product(cls.BOARD_SHAPE):
            if state[i, j, player_index] == 1:
                for p_x, p_y in cls.shoot(state, i, j):
                    partial_move = cls.null_move(state)
                    partial_move[i, j, player_index] = 0
                    partial_move[p_x, p_y, player_index] = 1
                    for t_x, t_y in cls.shoot(partial_move, p_x, p_y):
                        full_move = np.copy(partial_move)  # Don't use null_move because turn was already switched above
                        full_move[t_x, t_y, 2] = 1
                        moves.append(full_move)
        return moves

    @classmethod
    def get_legal_moves(cls, state):
        legal_moves = np.full(cls.MOVE_SHAPE, False)

        player_index = 0 if cls.is_player_1_turn(state) else 1
        for i, j in iter_product(cls.BOARD_SHAPE):
            if state[i, j, player_index] == 1:
                for p_x, p_y in cls.shoot(state, i, j):
                    partial_move = cls.null_move(state)
                    partial_move[i, j, player_index] = 0
                    partial_move[p_x, p_y, player_index] = 1
                    p_direction, p_distance = cls.parse(p_x - i, p_y - j)
                    for t_x, t_y in cls.shoot(partial_move, p_x, p_y):
                        t_direction, t_distance = cls.parse(t_x - p_x, t_y - p_y)
                        legal_moves[i, j, p_direction, p_distance, t_direction, t_distance] = True
        return legal_moves

    @classmethod
    def get_ruleset(cls):
        return f'{cls.ROWS}x{cls.COLUMNS}'

    @staticmethod
    def parse(di, dj):
        direction = DIRECTIONS_8.index((np.sign(di), np.sign(dj)))
        distance = np.maximum(np.abs(di), np.abs(dj)) - 1
        return direction, distance

    @classmethod
    def is_over(cls, state):
        return len(cls.get_possible_moves(state)) == 0

    @classmethod
    def get_winner(cls, state):
        if not cls.is_over(state):
            raise Exception('Game is not over!')
        return -1 if cls.is_player_1_turn(state) else 1
