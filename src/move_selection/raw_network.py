from time import sleep
from move_selection.move_chooser import MoveChooser


class RawNetwork(MoveChooser):
    def __init__(self, GameClass, starting_position=None, network=None, optimal=False, delay=2):
        super().__init__(GameClass=GameClass, starting_position=starting_position)
        if network is None:
            raise ValueError('Network must be provided!')
        self.network = network
        self.delay = delay
        self.optimal = optimal

    def start(self):
        self.network.initialize()

    def choose_move(self, return_distribution=False):
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        if self.delay > 0:
            sleep(self.delay)

        is_ai_player_1 = self.GameClass.is_player_1_turn(self.position)
        chosen_moves = []

        while self.GameClass.is_player_1_turn(self.position) == is_ai_player_1:
            self.position, distribution = self.network.choose_move(self.position, return_distribution=True,
                                                                   optimal=self.optimal)
            chosen_moves.append((self.position, distribution)
                                if return_distribution else self.position)
        return chosen_moves
