from src.games.Game import Game
import numpy as np
from src.utils.Utils import iter_product


class Amazons(Game):
    WHITE_STARTING_BOARD = np.array([[0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0]])
    BLACK_STARTING_BOARD = np.rot90(WHITE_STARTING_BOARD)
    STARTING_STATE = np.stack([WHITE_STARTING_BOARD,
                               BLACK_STARTING_BOARD,
                               np.zeros_like(WHITE_STARTING_BOARD),
                               np.ones_like(WHITE_STARTING_BOARD)], axis=-1)

    STATE_SHAPE = STARTING_STATE.shape  # 6, 6, 4
    BOARD_SHAPE = STATE_SHAPE[:-1]  # 6, 6
    ROWS, COLUMNS = BOARD_SHAPE
    BOARD_LENGTH = BOARD_SHAPE[0]
    FEATURE_COUNT = STATE_SHAPE[-1]  # 3
    REPRESENTATION_LETTERS = ['W', 'B', 'X']
    REPRESENTATION_FILES = ['white_circle', 'black_circle', 'red_circle']
    CLICKS_PER_MOVE = 3

    def __init__(self, state=STARTING_STATE):
        super().__init__()
        self.state = np.copy(state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = np.copy(state)

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

    def draw(self, canvas=None, move_prompt=False):
        if canvas is None:
            matrix = Amazons.get_human_readable_representation(self.state)
            for i in range(matrix.shape[0]):
                print(' '.join(matrix[i, :]))
        else:
            raise NotImplementedError()

    @classmethod
    def get_representation_shape(cls):
        return cls.STATE_SHAPE

    @classmethod
    def get_human_readable_representation(cls, state):
        representation = np.full(cls.BOARD_SHAPE, '_', dtype=str)
        for i in range(cls.FEATURE_COUNT - 1):  # -1 to exclude the turn information
            representation[state[:, :, i] == 1] = cls.REPRESENTATION_LETTERS[i]
        return representation

    @classmethod
    def get_starting_state(cls):
        return np.copy(cls.STARTING_STATE)

    @staticmethod
    def is_valid(i, j):
        return 0 <= i < Amazons.BOARD_LENGTH and 0 <= j < Amazons.BOARD_LENGTH

    @staticmethod
    def shoot(state, i, j):
        targets = []
        for v_x, v_y in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for t in range(1, Amazons.BOARD_LENGTH):
                p_x, p_y = i + t * v_x, j + t * v_y
                if Amazons.is_valid(p_x, p_y) and np.all(state[p_x, p_y, :3] == 0):
                    targets.append((p_x, p_y))
                else:
                    break
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
    def is_over(cls, state):
        return len(cls.get_possible_moves(state)) == 0

    @classmethod
    def get_winner(cls, state):
        if not cls.is_over(state):
            raise Exception('Game is not over!')
        return -1 if cls.is_player_1_turn(state) else 1
