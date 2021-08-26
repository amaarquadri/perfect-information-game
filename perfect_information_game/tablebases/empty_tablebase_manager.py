import numpy as np
from perfect_information_game.tablebases import AbstractTablebaseManager


class EmptyTablebaseManager(AbstractTablebaseManager):
    """
    Implements an empty tablebase manager.
    """
    def update_tablebase_list(self):
        pass

    def query_position(self, state, outcome_only=False):
        if self.GameClass.is_over(state):
            return (self.GameClass.get_winner(state), 0) if outcome_only \
                else (None, self.GameClass.get_winner(state), 0)

        return (np.nan, np.nan) if outcome_only else (None, np.nan, np.nan)
