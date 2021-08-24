from abc import ABC, abstractmethod
from os import listdir
import pickle
from perfect_information_game.utils import get_training_path
from perfect_information_game.utils import choose_random


class AbstractTablebaseManager(ABC):
    """
    Abstract superclass of tablebase managers.
    Each tablebase manager manages a set of tablebase files which are located in
    {get_training_path(GameClass)}/tablebases/{descriptor}.pickle

    Each file contains information for all positions that have the given descriptor as determined by
    GameClass.get_position_descriptor

    Each file is a pickled dictionary.
    The keys are bytes objects which represent a position as defined by GameClass.parse_board_bytes.
    The values are bytes objects which represent (move_data, outcome, terminal_distance) tuples as defined by
    GameClass.parse_move_bytes.
    """
    def __init__(self, GameClass):
        self.GameClass = GameClass

        # dictionary mapping descriptors to tablebases
        self.tablebases = {}

        self.available_tablebases = []
        self.update_tablebase_list()

    def update_tablebase_list(self):
        tablebases = [file[:-len('.pickle')] for file in listdir(f'{get_training_path(self.GameClass)}/tablebases')
                      if file.endswith('.pickle')]
        self.available_tablebases.extend([tablebase for tablebase in tablebases
                                         if tablebase not in self.available_tablebases])

    def ensure_loaded(self, descriptor):
        if descriptor not in self.tablebases:
            if descriptor not in self.available_tablebases:
                self.update_tablebase_list()
                if descriptor not in self.available_tablebases:
                    raise NotImplementedError(f'No tablebase available for descriptor = {descriptor}')
            with open(f'{get_training_path(self.GameClass)}/tablebases/{descriptor}.pickle', 'rb') as file:
                self.tablebases[descriptor] = pickle.load(file)

    @abstractmethod
    def query_position(self, state, outcome_only=False):
        pass

    def get_random_endgame(self, descriptor, condition=None):
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
                                       self.GameClass.parse_move_bytes(move_bytes)[1] == outcome
                                       and not self.GameClass.is_over(self.GameClass.parse_board_bytes(board_bytes)))
