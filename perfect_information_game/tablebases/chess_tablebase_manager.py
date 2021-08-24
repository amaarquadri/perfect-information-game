import numpy as np
from perfect_information_game.tablebases import SymmetryTransform
from perfect_information_game.tablebases import AbstractTablebaseManager, get_verified_chess_subclass


class ChessTablebaseManager(AbstractTablebaseManager):
    """
    Each tablebase has a descriptor, in a form such as KQkn (king and queen vs king and knight).
    Only the symmetrically unique variants of each position are stored in the tablebases.
    """
    def __init__(self, GameClass=None):
        super().__init__(get_verified_chess_subclass(GameClass))

    def query_position(self, state, outcome_only=False):
        """
        Checks if the given position is in one of the existing tablebases.
        Returns a tuple containing the state after the optimal move has been made, the game's outcome,
        and the terminal distance.

        If the position is not available in the tablebases, then (None, np.nan, np.nan) will be returned.
        If the position is a draw by insufficient material, then (None, 0, 0) will be returned.

        :param state:
        :param outcome_only: If True, then the state after the move has been made
                             will not be included in the returned tuple.
        """
        if np.any(state[:, :, -2] == 1):
            # any positions with en passant or castling are excluded from the tablebase
            return (np.nan, np.nan) if outcome_only else (None, np.nan, np.nan)

        symmetry_transform = SymmetryTransform(self.GameClass, state)
        transformed_state = symmetry_transform.transform_state(state)

        descriptor = self.GameClass.get_position_descriptor(transformed_state)

        if descriptor in self.GameClass.DRAWING_DESCRIPTORS:
            return (0, 0) if outcome_only else (None, 0, 0)

        if descriptor not in self.available_tablebases:
            return (np.nan, np.nan) if outcome_only else (None, np.nan, np.nan)

        self.ensure_loaded(descriptor)
        tablebase = self.tablebases[descriptor]
        move_bytes = tablebase[self.GameClass.encode_board_bytes(transformed_state)]
        (start_i, start_j, end_i, end_j), outcome, terminal_distance = self.GameClass.parse_move_bytes(move_bytes)
        outcome = symmetry_transform.transform_outcome(outcome)
        if outcome_only:
            return outcome, terminal_distance

        if terminal_distance == 0:
            return None, outcome, terminal_distance

        transformed_move_state = self.GameClass.apply_from_to_move(transformed_state, start_i, start_j, end_i, end_j)
        move_state = symmetry_transform.untransform_state(transformed_move_state)
        return move_state, outcome, terminal_distance

    def get_random_endgame(self, descriptor, condition=None):
        state = super().get_random_endgame(descriptor, condition)
        return SymmetryTransform.random(self.GameClass, descriptor).transform_state(state)
