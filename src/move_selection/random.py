from src.utils.utils import choose_random
from time import sleep


class RandomMoveChooser:
    def __init__(self, GameClass, delay=1):
        self.GameClass = GameClass
        self.delay = delay

    def start(self):
        pass

    def terminate(self):
        pass

    def choose_move(self, position):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        sleep(self.delay)
        return choose_random(self.GameClass.get_possible_moves(position))
