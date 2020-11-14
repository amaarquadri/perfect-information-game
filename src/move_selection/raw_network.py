from time import sleep
from src.move_selection.move_chooser import MoveChooser


class RawNetwork(MoveChooser):
    def __init__(self, network, starting_position=None, optimal=False, delay=2):
        super().__init__(GameClass=None, starting_position=starting_position)
        self.network = network
        self.delay = delay
        self.optimal = optimal

    def start(self):
        self.network.initialize()

    def choose_move(self, return_distribution=False):
        if self.delay > 0:
            sleep(self.delay)
        self.position = self.network.choose_move(self.position, return_distribution, self.optimal)
        return self.position
