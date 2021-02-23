from .game import Game
import numpy as np
from utils.utils import iter_product, DIRECTIONS_8


class Othello(Game):
    PLAYER_1_STARTING_BOARD = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]])
    PLAYER_2_STARTING_BOARD = np.rot90(PLAYER_1_STARTING_BOARD)
    STARTING_STATE = np.stack([PLAYER_1_STARTING_BOARD, PLAYER_2_STARTING_BOARD,
                               np.ones_like(PLAYER_1_STARTING_BOARD)], axis=-1).astype(np.uint8)

    STATE_SHAPE = STARTING_STATE.shape  # 8, 8, 3
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 8, 8, 3
    BOARD_SHAPE = (ROWS, COLUMNS)  # 8, 8
    MOVE_SHAPE = BOARD_SHAPE  # 8, 8
    REPRESENTATION_LETTERS = ['b', 'w']
    REPRESENTATION_FILES = ['dark_square', 'black_circle_dark_square', 'white_circle_dark_square']
    CLICKS_PER_MOVE = 1

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        # check for forced pass due to no legal moves
        if np.all(np.logical_not(self.get_legal_moves(self.state))):
            self.state = self.null_move(self.state)
            return

        i, j = clicks[0]
        if np.any(self.state[i, j, :2] == 1):
            raise ValueError('Illegal Move: square is occupied!')

        friendly_index = 0 if self.is_player_1_turn(self.state) else 1
        enemy_index = 1 - friendly_index
        flip_squares = np.full(self.BOARD_SHAPE, False)
        for di, dj, in DIRECTIONS_8:
            p_x, p_y = i + di, j + dj
            if not (self.is_valid(p_x, p_y) and self.state[p_x, p_y, enemy_index] == 1):
                continue
            p_x += di
            p_y += dj

            while self.is_valid(p_x, p_y) and self.state[p_x, p_y, enemy_index] == 1:
                p_x += di
                p_y += dj

            if self.is_valid(p_x, p_y) and self.state[p_x, p_y, friendly_index] == 1:
                # success, mark all squares between i, j and p_x, p_y (not including endpoints)
                p_x -= di
                p_y -= dj
                while not (p_x == i and p_y == j):
                    flip_squares[p_x, p_y] = True
                    p_x -= di
                    p_y -= dj

        if np.any(flip_squares):
            new_state = self.null_move(self.state)
            player_piece = [enemy_index, friendly_index]
            new_state[i, j, :2] = player_piece
            new_state[flip_squares, :2] = player_piece
            self.state = new_state
        else:
            raise ValueError('Illegal Move: no pieces are captured by that move!')

    @classmethod
    def get_possible_moves(cls, state):
        moves = []

        friendly_index = 0 if cls.is_player_1_turn(state) else 1
        enemy_index = 1 - friendly_index
        player_piece = [enemy_index, friendly_index]
        for i, j in iter_product(cls.BOARD_SHAPE):
            if np.any(state[i, j, :2] == 1):
                continue

            flip_squares = np.full(cls.BOARD_SHAPE, False)
            for di, dj, in DIRECTIONS_8:
                p_x, p_y = i + di, j + dj
                if not (cls.is_valid(p_x, p_y) and state[p_x, p_y, enemy_index] == 1):
                    continue
                p_x += di
                p_y += dj

                while cls.is_valid(p_x, p_y) and state[p_x, p_y, enemy_index] == 1:
                    p_x += di
                    p_y += dj

                if cls.is_valid(p_x, p_y) and state[p_x, p_y, friendly_index] == 1:
                    # success, mark all squares between i, j and p_x, p_y (not including endpoints)
                    p_x -= di
                    p_y -= dj
                    while not (p_x == i and p_y == j):
                        flip_squares[p_x, p_y] = True
                        p_x -= di
                        p_y -= dj

            if np.any(flip_squares):
                move = cls.null_move(state)
                move[i, j, :2] = player_piece
                move[flip_squares, :2] = player_piece
                moves.append(move)

        if len(moves) == 0:
            pass_move = cls.null_move(state)
            moves.append(pass_move)
        return moves

    @classmethod
    def get_legal_moves(cls, state):
        legal_moves = np.full(cls.BOARD_SHAPE, False)

        friendly_index = 0 if cls.is_player_1_turn(state) else 1
        enemy_index = 1 - friendly_index
        for i, j in iter_product(cls.BOARD_SHAPE):
            if np.any(state[i, j, :2] == 1):
                continue

            for di, dj, in DIRECTIONS_8:
                p_x, p_y = i + di, j + dj
                if not (cls.is_valid(p_x, p_y) and state[p_x, p_y, enemy_index] == 1):
                    continue
                p_x += di
                p_y += dj

                while cls.is_valid(p_x, p_y) and state[p_x, p_y, enemy_index] == 1:
                    p_x += di
                    p_y += dj

                if cls.is_valid(p_x, p_y) and state[p_x, p_y, friendly_index] == 1:
                    legal_moves[i, j] = True
                    break

        return legal_moves

    @classmethod
    def is_over(cls, state):
        return np.all(np.logical_not(cls.get_legal_moves(state))) and \
               np.all(np.logical_not(cls.get_legal_moves(cls.null_move(state))))

    @classmethod
    def get_winner(cls, state):
        player_1_points = np.sum(state[:, :, 0])
        player_2_points = np.sum(state[:, :, 1])
        if player_1_points > player_2_points:
            return 1
        if player_2_points > player_1_points:
            return -1
        return 0
