import numpy as np
from games.chess import Chess
from utils.utils import iter_product


class SymmetryTransform:
    # noinspection PyChainedComparisons
    UNIQUE_SQUARE_INDICES = [(i, j) for i, j in iter_product(Chess.BOARD_SHAPE) if i < 4 and j < 4 and i <= j]

    @staticmethod
    def requires_transform(i, j):
        # noinspection PyChainedComparisons
        return not (i < 4 and j < 4 and i <= j)

    def __init__(self, state, attacking_slice):
        i, j = Chess.get_king_pos(state, attacking_slice)
        if i < 4:
            self.flip_i = False
        else:
            self.flip_i = True
            i = 7 - i
        if j < 4:
            self.flip_j = False
        else:
            self.flip_j = True
            j = 7 - j
        self.flip_diagonal = not (i <= j)

    def transform_coordinates(self, i, j):
        if self.flip_i:
            i = 7 - i
        if self.flip_j:
            j = 7 - j
        if self.flip_diagonal:
            i, j = j, i
        return i, j

    def transform_state(self, state):
        if self.flip_i:
            state = np.flip(state, axis=0)
        if self.flip_j:
            state = np.flip(state, axis=1)
        if self.flip_diagonal:
            state = np.rot90(np.flip(state, axis=1), axes=(0, 1))
        return state
