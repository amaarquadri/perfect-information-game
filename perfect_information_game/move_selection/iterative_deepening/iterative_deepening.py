import numpy as np
from perfect_information_game.move_selection import MoveChooser
from perfect_information_game.move_selection.iterative_deepening import DeepeningNode


class IterativeDeepening(MoveChooser):
    def __init__(self, GameClass, starting_position=None, depth=3):
        super().__init__(GameClass, starting_position)
        self.root = DeepeningNode(GameClass, starting_position)
        self.depth = depth

    def report_user_move(self, user_chosen_position):
        if self.root.terminal:
            raise ValueError('Game is over!')

        if self.root.children is None:
            self.root.deepen()

        for child_node in self.root.children:
            if np.all(user_chosen_position == child_node.state):
                self.root = child_node
                self.position = user_chosen_position
                break
        else:
            raise ValueError('Invalid move!')

    def reset(self):
        super().reset()
        self.root = DeepeningNode(self.GameClass, self.position)

    def choose_move(self, return_distribution=False):
        if return_distribution:
            raise NotImplementedError

        if self.root.terminal:
            raise ValueError('Game is over!')

        while self.root.get_depth() < self.depth:
            self.root.deepen()
            print(self.root.get_depth())

        self.root = self.root.children[0]
        return [self.root.state]
