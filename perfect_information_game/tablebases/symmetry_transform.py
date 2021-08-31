import numpy as np
from perfect_information_game.games import Chess
from perfect_information_game.utils import iter_product
from perfect_information_game.tablebases import get_verified_chess_subclass


class SymmetryTransform:
    # noinspection PyChainedComparisons
    PAWNLESS_UNIQUE_SQUARE_INDICES = [(i, j) for i, j in iter_product(Chess.BOARD_SHAPE)
                                      if i < 4 and j < 4 and i <= j]
    UNIQUE_SQUARE_INDICES = [(i, j) for i, j in iter_product(Chess.BOARD_SHAPE) if j < 4]

    def __init__(self, GameClass, state):
        self.GameClass = get_verified_chess_subclass(GameClass)
        self.flip_colors = self.flip_i = self.flip_j = self.flip_diagonal = False

        if self.should_swap_colours(state):
            # black is attacking, so switch white and black
            self.flip_colors = True
            i, j = self.GameClass.get_king_pos(state, self.GameClass.BLACK_SLICE)
            i = self.GameClass.ROWS - 1 - i
        else:
            i, j = self.GameClass.get_king_pos(state, self.GameClass.WHITE_SLICE)

        pawnless = np.all(state[:, :, self.GameClass.WHITE_PAWN] == 0) and \
            np.all(state[:, :, self.GameClass.BLACK_PAWN] == 0)

        if pawnless and not (i < 4):
            self.flip_i = True
            i = self.GameClass.ROWS - 1 - i
        if not (j < 4):  # horizontal flipping can be done, even with pawns
            self.flip_j = True
            j = self.GameClass.COLUMNS - 1 - j
        if pawnless and not (i <= j):
            self.flip_diagonal = True

    def should_swap_colours(self, state):
        heuristic = self.GameClass.heuristic(state)
        if heuristic > 0:
            # white is up in material, so don't swap colours
            return False
        if heuristic < 0:
            # black is up in material, so swap colours
            return True
        # compare the number of pawns on each rank, from most advanced to least advanced pawns
        # no need to check second rank pawns, because if everything else is equal they must be equal too
        for rank in range(7, 2, -1):
            if np.sum(state[rank - 1, :, self.GameClass.BLACK_PAWN]) > \
                    np.sum(state[8 - rank, :, self.GameClass.WHITE_PAWN]):
                # black has more pawns than white on this rank, so swap colours
                return True
        return False

    @staticmethod
    def identity(GameClass):
        identity = SymmetryTransform(GameClass, GameClass.STARTING_STATE)
        identity.flip_colors = identity.flip_i = identity.flip_j = identity.flip_diagonal = False
        return identity

    @staticmethod
    def random(GameClass, descriptor):
        """
        Returns a random symmetry transform for the given descriptor.
        """
        random = SymmetryTransform.identity(GameClass)
        pawnless = 'p' not in descriptor and 'P' not in descriptor

        random.flip_colors = np.random.random() < 0.5
        random.flip_j = np.random.random() < 0.5
        if pawnless:
            random.flip_i = np.random.random() < 0.5
            random.flip_diagonal = np.random.random() < 0.5
        return random

    def is_identity(self):
        return not self.flip_colors and not self.flip_i and not self.flip_j and not self.flip_diagonal

    def transform_state(self, state):
        if self.flip_colors:
            state = self.flip_state_colors(self.GameClass, state)
        if self.flip_i:
            state = self.flip_state_i(state)
        if self.flip_j:
            state = self.flip_state_j(state)
        if self.flip_diagonal:
            state = self.flip_state_diagonal(state)
        return state

    def untransform_state(self, state):
        # since all transform_funcs are their own inverses, we can just run through them in reverse
        if self.flip_diagonal:
            state = self.flip_state_diagonal(state)
        if self.flip_j:
            state = self.flip_state_j(state)
        if self.flip_i:
            state = self.flip_state_i(state)
        if self.flip_colors:
            state = self.flip_state_colors(self.GameClass, state)
        return state

    def transform_outcome(self, outcome):
        return -outcome if self.flip_colors else outcome

    @staticmethod
    def flip_state_colors(GameClass, state):
        special_layers = np.copy(state[..., -2:])
        special_layers[..., -1] = 1 - special_layers[..., -1]  # flip whose turn it is
        new_state = np.concatenate((state[..., GameClass.BLACK_SLICE], state[..., GameClass.WHITE_SLICE],
                                    special_layers),
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
