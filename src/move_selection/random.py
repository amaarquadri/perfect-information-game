from src.utils.utils import choose_random


class RandomMoveChooser:
    def __init__(self, GameClass):
        self.GameClass = GameClass

    def start(self):
        pass

    def terminate(self):
        pass

    def choose_move(self, position):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        return choose_random(self.GameClass.get_possible_moves(position))
