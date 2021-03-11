import numpy as np
from move_selection.move_chooser import MoveChooser


class MiniMax(MoveChooser):
    """
    Implementation of MiniMax with alpha-beta pruning.
    """

    def __init__(self, GameClass, starting_position=None, heuristic_func=None, depth=5):
        """
        The heuristic approximation should be positive if player 1 is winning,
        negative if player 2 is winning, and 0 if it is a draw.

        :param heuristic_func: Take's positions for the game and returns a heuristic approximation.
        :param depth: 
        """
        super().__init__(GameClass, starting_position)
        self.heuristic_func = heuristic_func if heuristic_func is not None else GameClass.heuristic
        self.depth = depth

    @staticmethod
    def from_network(GameClass, starting_position=None, network=None, depth=5):
        if network is None:
            raise ValueError('Network must be provided!')

        def heuristic_func(state):
            _, evaluation = network.predict(state[np.newaxis, ...])[0]
            return evaluation

        class NetworkMiniMax(MiniMax):
            def __init__(self):
                super().__init__(GameClass, starting_position, heuristic_func, depth)

            def start(self):
                network.initialize()
        return NetworkMiniMax()

    @staticmethod
    def solver(GameClass, starting_position=None):
        return MiniMax(GameClass, starting_position=starting_position, depth=np.inf)

    def choose_move(self, return_distribution=False):
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        is_maximizing = self.GameClass.is_player_1_turn(self.position)
        best_move = None
        best_heuristic = -np.inf if is_maximizing else np.inf
        heuristics = []

        for move in self.GameClass.get_possible_moves(self.position):
            heuristic = self.evaluate_position_recursive(move, self.depth - 1, not is_maximizing, best_heuristic)
            heuristics.append(heuristic)

            if (is_maximizing and heuristic > best_heuristic) or (not is_maximizing and heuristic < best_heuristic):
                best_heuristic = heuristic
                best_move = move

        self.position = best_move

        if return_distribution:
            # create an exponentially scaled distribution based on the heuristic values
            distribution = np.exp(heuristics)
            distribution /= np.sum(distribution)
            return self.position, distribution
        else:
            return self.position

    def evaluate_position_recursive(self, position, depth, is_maximizing, value_to_beat):
        if self.GameClass.is_over(position):
            return self.GameClass.get_winner(position)

        if depth == 0:
            return self.heuristic_func(position)

        best_heuristic = -np.inf if is_maximizing else np.inf
        for child_position in self.GameClass.get_possible_moves(position):
            heuristic = self.evaluate_position_recursive(child_position, depth - 1, not is_maximizing, best_heuristic)

            if is_maximizing and heuristic > best_heuristic:
                if heuristic > value_to_beat:
                    # prune
                    return heuristic
                best_heuristic = heuristic
            if not is_maximizing and heuristic < best_heuristic:
                if heuristic < value_to_beat:
                    # prune
                    return heuristic
                best_heuristic = heuristic

        return best_heuristic
