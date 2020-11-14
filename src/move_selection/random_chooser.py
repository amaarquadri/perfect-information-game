from src.utils.utils import choose_random
from time import sleep
import numpy as np


class RandomMoveChooser:
    def __init__(self, GameClass, delay=1):
        self.GameClass = GameClass
        self.delay = delay

    def start(self):
        pass

    def terminate(self):
        pass

    def choose_move(self, position, return_distribution=False):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        sleep(self.delay)
        moves = self.GameClass.get_possible_moves(position)
        chosen_move = choose_random(moves)
        return (chosen_move, np.full_like(moves, 1 / len(moves))) if return_distribution else chosen_move

    def generate_random_game(self, max_moves=np.inf):
        training_data = []
        position = self.GameClass.STARTING_STATE
        while not self.GameClass.is_over(position):
            if len(training_data) == max_moves:
                return training_data, 0
            new_position, distribution = self.choose_move(position, return_distribution=True)
            training_data.append((position, distribution))
            position = new_position
        return training_data, self.GameClass.get_winner(position)
