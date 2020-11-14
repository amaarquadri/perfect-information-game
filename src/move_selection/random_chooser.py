from src.utils.utils import choose_random
from time import sleep
import numpy as np
from src.move_selection.move_chooser import MoveChooser


class RandomMoveChooser(MoveChooser):
    def __init__(self, GameClass, starting_position=None, delay=1):
        super().__init__(GameClass, starting_position)
        self.delay = delay

    def choose_move(self, return_distribution=False):
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        if self.delay > 0:
            sleep(self.delay)
        moves = self.GameClass.get_possible_moves(self.position)
        self.position = choose_random(moves)
        return (self.position, np.full_like(moves, 1 / len(moves))) if return_distribution else self.position
