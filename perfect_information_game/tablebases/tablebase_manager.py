import pickle
from os import listdir
import numpy as np
from perfect_information_game.tablebases import SymmetryTransform
from perfect_information_game.utils import choose_random, get_training_path
from perfect_information_game.tablebases import get_verified_chess_subclass


class TablebaseManager:
    """
    Each tablebase has a descriptor, in a form such as KQkn (king and queen vs king and knight).
    The tablebases are stored in {get_training_path(GameClass)}/tablebases/{descriptor}.pickle

    Each tablebase consists of a dictionary that maps board_bytes to move_bytes.
    move_bytes can be converted to and from this tuple: (outcome, start_i, start_j, target_i, target_j, distance).
    Only the symmetric variants of each position are stored in the tablebases.
    """

    @staticmethod
    def encode_move_bytes(outcome, start_i, start_j, end_i, end_j, terminal_distance):
        if terminal_distance < 0:
            raise ValueError(f'terminal_distance < 0: {terminal_distance}')
        if np.isinf(terminal_distance):
            terminal_distance = 2 ** 10 - 1  # maximum value in 10 bits
        elif terminal_distance >= 2 ** 10 - 1:
            print(f'Warning: terminal_distance of {terminal_distance} is too large. '
                  f'Replacing with max value of {2 ** 10 - 2}')
            # replace with 2^10 - 2 because 2^10 - 1 is reserved for infinity
            terminal_distance = 2 ** 10 - 2
        outcome += 1  # remap from -1, 0, 1 to 0, 1, 2

        move_bytes = [outcome * 2 ** 6 + start_i * 2 ** 3 + start_j,
                      end_i * 2 ** 5 + end_j * 2 ** 2 + terminal_distance // (2 ** 8),
                      terminal_distance % (2 ** 8)]
        return bytes(move_bytes)

    @staticmethod
    def parse_move_bytes(move_bytes):
        move_bytes = list(move_bytes)  # convert back to list of integers
        outcome = move_bytes[0] // (2 ** 6)
        outcome -= 1  # map from 0, 1, 2 back to -1, 0, 1

        start_i = (move_bytes[0] % (2 ** 6)) // (2 ** 3)
        start_j = move_bytes[0] % (2 ** 3)
        end_i = move_bytes[1] // (2 ** 5)
        end_j = (move_bytes[1] % (2 ** 5)) // (2 ** 2)

        terminal_distance = (move_bytes[1] % (2 ** 2)) * (2 ** 8) + move_bytes[2]
        if terminal_distance == 2 ** 10 - 1:
            terminal_distance = np.inf
        elif terminal_distance == 2 ** 10 - 2:
            print(f'Warning: terminal distance of {terminal_distance} may be incorrect due to overflow!')

        return outcome, start_i, start_j, end_i, end_j, terminal_distance

    def __init__(self, GameClass=None):
        self.GameClass = get_verified_chess_subclass(GameClass)

        # dictionary mapping descriptors to tablebases
        self.tablebases = {}

        self.available_tablebases = [file[:-len('.pickle')]
                                     for file in listdir(f'{get_training_path(self.GameClass)}/tablebases')
                                     if file.endswith('.pickle') and '_nodes' not in file]

    def update_tablebase_list(self):
        tablebases = [file[:-len('.pickle')] for file in listdir(f'{get_training_path(self.GameClass)}/tablebases')
                      if file.endswith('.pickle') and '_nodes' not in file]
        self.available_tablebases.extend([tablebase for tablebase in tablebases
                                         if tablebase not in self.available_tablebases])

    def ensure_loaded(self, descriptor):
        if descriptor not in self.tablebases:
            with open(f'{get_training_path(self.GameClass)}/tablebases/{descriptor}.pickle', 'rb') as file:
                self.tablebases[descriptor] = pickle.load(file)

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
        outcome, start_i, start_j, end_i, end_j, terminal_distance = TablebaseManager.parse_move_bytes(move_bytes)
        outcome = symmetry_transform.transform_outcome(outcome)
        if outcome_only:
            return outcome, terminal_distance

        if terminal_distance == 0:
            return None, outcome, terminal_distance

        transformed_move_state = self.GameClass.apply_from_to_move(transformed_state, start_i, start_j, end_i, end_j)
        move_state = symmetry_transform.untransform_state(transformed_move_state)
        return move_state, outcome, terminal_distance

    def get_random_endgame(self, descriptor, condition=None):
        if descriptor not in self.available_tablebases:
            raise NotImplementedError(f'No tablebase available for descriptor = {descriptor}')

        self.ensure_loaded(descriptor)
        tablebase = self.tablebases[descriptor]

        if condition is None:
            allowed_board_bytes = list(tablebase.keys())
        else:
            allowed_board_bytes = [board_bytes for board_bytes, move_bytes in tablebase.items()
                                   if condition(board_bytes, move_bytes)]
            if len(allowed_board_bytes) == 0:
                return None
        return self.GameClass.parse_board_bytes(choose_random(allowed_board_bytes))

    def get_random_endgame_with_outcome(self, descriptor, outcome):
        return self.get_random_endgame(descriptor,
                                       lambda board_bytes, move_bytes:
                                       self.parse_move_bytes(move_bytes)[0] == outcome
                                       and not self.GameClass.is_over(self.GameClass.parse_board_bytes(board_bytes)))
