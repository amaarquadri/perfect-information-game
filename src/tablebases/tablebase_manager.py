import pickle
from os import listdir
import numpy as np
from games.chess import Chess, encode_fen, parse_fen, PIECE_LETTERS
from tablebases.symmetry_transform import SymmetryTransform


class TablebaseManager:
    """
    Each tablebase has a descriptor, in a form such as KQkn (king and queen vs king and knight).
    The tablebases are stored in chess_tablebases/{descriptor}.pickle

    Each tablebase consists of a dictionary that maps fens to (start_i, start_j, target_i, target_j, outcome, distance).
    Only the symmetric variants of each position are stored in the tablebases.
    """
    DRAWING_DESCRIPTORS = ['Kk', 'KBk', 'KNk']
    AVAILABLE_TABLEBASES = [file[:len('.pickle')] for file in listdir('chess_tablebases') if file.endswith('.pickle')]

    def __init__(self):
        # dictionary mapping descriptors to tablebases
        self.tablebases = {}

    def query_position(self, state, outcome_only=False):
        """
        Checks if the given position is in one of the existing tablebases.
        Returns a tuple containing the state after the optimal move has been made, the game's outcome,
        and the terminal distance.

        If the position is not available in the tablebases, then (None, np.nan, np.nan) will be returned.
        If the position is a draw by insufficient material, then (None, 0, 0) will be returned.

        :param state:
        :param outcome_only: If True, then only the state after the move has been made will not be calculated.
        """
        if np.any(state[:, :, -2] == 1):
            return (np.nan, np.nan) if outcome_only else (None, np.nan, np.nan)

        symmetry_transform = SymmetryTransform(state)
        transformed_state = symmetry_transform.transform_state(state)

        descriptor = self.get_position_descriptor(transformed_state)

        if descriptor in self.DRAWING_DESCRIPTORS:
            return (0, 0) if outcome_only else (None, 0, 0)

        if descriptor not in self.AVAILABLE_TABLEBASES:
            return (np.nan, np.nan) if outcome_only else (None, np.nan, np.nan)

        if descriptor not in self.tablebases:
            with open(f'chess_tablebases/{descriptor}.pickle', 'rb') as file:
                self.tablebases[descriptor] = pickle.load(file)

        tablebase = self.tablebases[descriptor]
        transformed_move_fen, outcome, distance = tablebase[encode_fen(transformed_state)]
        if outcome_only:
            return outcome, distance

        move_state = symmetry_transform.untransform_state(parse_fen(transformed_move_fen))
        return move_state, outcome, distance

    @staticmethod
    def get_position_descriptor(state):
        piece_counts = [np.sum(state[:, :, i] == 1) for i in range(12)]
        return ''.join([piece_count * letter for piece_count, letter in zip(piece_counts, PIECE_LETTERS)])
