import numpy as np


class DeepeningNode:
    def __init__(self, GameClass, state, heuristic_func=None):
        if state is None:
            state = GameClass.STARTING_STATE
        if heuristic_func is None:
            heuristic_func = GameClass.heuristic

        self.GameClass = GameClass
        self.state = state
        self.heuristic_func = heuristic_func
        self.is_maximizing = GameClass.is_player_1_turn(state)
        if GameClass.is_over(state):
            outcome = GameClass.is_over()
            self.heuristic = 0 if outcome == 0 else outcome * np.inf
            self.terminal = True
        else:
            self.heuristic = heuristic_func(state)
            self.terminal = False
        self.children = None

    def sort_children(self):
        self.children = sorted(self.children, key=lambda move: move.heuristic, reverse=self.is_maximizing)

    def deepen(self, value_to_beat=None):
        if self.terminal:
            return self.heuristic

        if self.children is None:
            # this is currently a leaf node, so deepen one final layer, then return
            moves = self.GameClass.get_possible_moves(self.state)
            self.children = [DeepeningNode(self.GameClass, move, self.heuristic_func) for move in moves]
            self.sort_children()
            self.heuristic = self.children[0].heuristic
            return self.heuristic

        best_heuristic = -np.inf if self.is_maximizing else np.inf
        if value_to_beat is None:
            value_to_beat = -best_heuristic
        for child in self.children:
            heuristic = child.deepen(value_to_beat=best_heuristic)

            if self.is_maximizing and heuristic > best_heuristic:
                if heuristic > value_to_beat:
                    # prune
                    self.sort_children()
                    return heuristic
                best_heuristic = heuristic
            if not self.is_maximizing and heuristic < best_heuristic:
                if heuristic < value_to_beat:
                    # prune
                    self.sort_children()
                    return heuristic
                best_heuristic = heuristic
        self.sort_children()
        self.heuristic = best_heuristic
        return best_heuristic

    def get_depth(self):
        """
        Note that this only searches down the first branch.
        So if the tree is unevenly deep then this will given an incorrect answer.

        :return:
        """
        if self.children is None:
            return 0
        return self.children[0].get_depth() + 1
