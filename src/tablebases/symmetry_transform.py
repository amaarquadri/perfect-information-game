import numpy as np
from utils.active_game import ActiveGame as GameClass
from games.chess import Chess
from utils.utils import iter_product


if not issubclass(GameClass, Chess):
    raise NotImplementedError('Cannot run SymmetryTransform for classes that do not inherit from Chess!')


class SymmetryTransform:
    # noinspection PyChainedComparisons
    PAWNLESS_UNIQUE_SQUARE_INDICES = [(i, j) for i, j in iter_product(GameClass.BOARD_SHAPE)
                                      if i < 4 and j < 4 and i <= j]
    UNIQUE_SQUARE_INDICES = [(i, j) for i, j in iter_product(GameClass.BOARD_SHAPE) if j < 4]

    def __init__(self, state):
        self.transform_funcs = []

        if GameClass.heuristic(state) < 0:
            # black is attacking, so switch white and black
            self.transform_funcs.append(SymmetryTransform.flip_state_colors)
            i, j = GameClass.get_king_pos(state, GameClass.BLACK_SLICE)
            i = GameClass.ROWS - 1 - i
        else:
            i, j = GameClass.get_king_pos(state, GameClass.WHITE_SLICE)

        pawnless = np.all(state[:, :, 5] == 0) and np.all(state[:, :, 11] == 0)

        if pawnless and not (i < 4):
            self.transform_funcs.append(SymmetryTransform.flip_state_i)
            i = GameClass.ROWS - 1 - i
        if not (j < 4):  # horizontal flipping can be done, even with pawns
            self.transform_funcs.append(SymmetryTransform.flip_state_j)
            j = GameClass.COLUMNS - 1 - j
        if pawnless and not (i <= j):
            self.transform_funcs.append(SymmetryTransform.flip_state_diagonal)

    @classmethod
    def identity(cls):
        identity = SymmetryTransform(GameClass.STARTING_STATE)
        identity.transform_funcs = []
        return identity

    def is_identity(self):
        return len(self.transform_funcs) == 0

    def transform_state(self, state):
        for transform_func in self.transform_funcs:
            state = transform_func(state)
        return state

    def untransform_state(self, state):
        for transform_func in self.transform_funcs[::-1]:
            state = transform_func(state)
        return state

    def transform_outcome(self, outcome):
        return -outcome if SymmetryTransform.flip_state_colors in self.transform_funcs else outcome

    @staticmethod
    def flip_state_colors(state):
        special_layers = np.copy(state[..., -2:])
        special_layers[..., -1] = 1 - special_layers[..., -1]
        new_state = np.concatenate((state[..., GameClass.BLACK_SLICE], state[..., GameClass.WHITE_SLICE], special_layers),
                                   axis=-1)
        # need to flip board vertically after flipping colours
        # this ensures that the pawns move in the correct directions
        return SymmetryTransform.flip_state_i(new_state)

    @staticmethod
    def flip_state_i(state):
        return np.flip(state, axis=0)

    @staticmethod
    def flip_state_j(state):
        return np.flip(state, axis=1)

    @staticmethod
    def flip_state_diagonal(state):
        return np.rot90(np.flip(state, axis=1), axes=(0, 1))
