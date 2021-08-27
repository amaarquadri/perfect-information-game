import numpy as np
from perfect_information_game.move_selection.mcts import AbstractNode
from perfect_information_game.tablebases import TablebaseException


class TablebaseNode(AbstractNode):
    def __init__(self, position, parent, GameClass, tablebase_manager, verbose=False):
        super().__init__(position, parent, GameClass, tablebase_manager, verbose=verbose)
        if self.fully_expanded:
            # the AbstractNode constructor already marked this node as fully_expanded because the game is over
            self.best_move = None
            self.terminal_distance = 0
        else:
            self.fully_expanded = True

            self.best_move, self.outcome, self.terminal_distance = tablebase_manager.query_position(position)
            if np.isnan(self.outcome):
                raise TablebaseException('Given position was not found in any existing tablebase!')

    @staticmethod
    def attempt_create(position, parent, GameClass, tablebase_manager, verbose=False, backup_factory=None):
        try:
            return TablebaseNode(position, parent, GameClass, tablebase_manager, verbose)
        except TablebaseException:
            if backup_factory is not None:
                return backup_factory()
            raise

    def get_evaluation(self):
        return self.outcome

    def count_expansions(self):
        return np.inf

    def set_fully_expanded(self, minimax_evaluation):
        raise NotImplementedError()

    def ensure_children(self):
        if self.children is None:
            self.children = [TablebaseNode(move, self, self.GameClass, self.tablebase_manager, self.verbose)
                             for move in self.GameClass.get_possible_moves(self.position)]

    def get_puct_heuristic_for_child(self, i):
        raise NotImplementedError()

    def choose_best_node(self, return_probability_distribution=False, optimal=True):
        if not optimal:
            print('Warning: non-optimal moves are not possible for TablebaseNode!')
        self.ensure_children()
        distribution = [1 if child.position == self.best_move else 0 for child in self.children]
        if np.sum(distribution) != 1:
            raise Exception('Inconsistent tablebase results!')
        idx = np.argmax(distribution)
        best_child = self.children[idx]
        return (best_child, distribution) if return_probability_distribution else best_child

    def expand(self):
        raise Exception('Node is fully expanded!')

    def choose_expansion_node(self, search_suboptimal=False):
        # no reason to expand any nodes since they are all in the tablebase
        return None

    def depth_to_end_game(self):
        return self.terminal_distance


