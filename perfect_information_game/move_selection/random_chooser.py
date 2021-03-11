from perfect_information_game.utils import choose_random
from time import sleep
import numpy as np
from perfect_information_game.move_selection import MoveChooser


class RandomMoveChooser(MoveChooser):
    def __init__(self, GameClass, starting_position=None, delay=1):
        super().__init__(GameClass, starting_position)
        self.delay = delay

    def choose_move(self, return_distribution=False):
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        if self.delay > 0:
            sleep(self.delay)

        is_ai_player_1 = self.GameClass.is_player_1_turn(self.position)
        chosen_moves = []

        while self.GameClass.is_player_1_turn(self.position) == is_ai_player_1:
            moves = self.GameClass.get_possible_moves(self.position)
            self.position = choose_random(moves)
            chosen_moves.append((self.position, np.full_like(moves, 1 / len(moves)))
                                if return_distribution else self.position)
        return chosen_moves
