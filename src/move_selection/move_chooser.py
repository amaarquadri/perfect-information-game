from abc import ABC, abstractmethod
import numpy as np


class MoveChooser(ABC):
    def __init__(self, GameClass, starting_position=None):
        self.GameClass = GameClass
        self.position = starting_position if starting_position is not None else GameClass.STARTING_STATE

    def start(self):
        pass

    def terminate(self):
        pass

    def report_user_move(self, user_chosen_position):
        """
        Accepts a single position that the user has made, and updates accordingly.

        :param user_chosen_position:
        :return:
        """
        if user_chosen_position not in self.GameClass.get_possible_moves(self.position):
            raise ValueError('Invalid move!')
        self.position = user_chosen_position

    @abstractmethod
    def choose_move(self, return_distribution=False):
        """
        Must update self.position

        @param return_distribution:
        @return: A list of all the positions that were made a s part of this move
        """
        pass

    def reset(self):
        self.position = np.copy(self.GameClass.STARTING_STATE)

    def generate_random_game(self, max_moves=np.inf, from_current_position=False):
        if not from_current_position:
            self.reset()
        training_data = []
        while not self.GameClass.is_over(self.position):
            if len(training_data) == max_moves:
                return training_data, 0
            position = self.position
            _, distribution = self.choose_move(return_distribution=True)
            training_data.append((position, distribution))
        return training_data, self.GameClass.get_winner(self.position)
