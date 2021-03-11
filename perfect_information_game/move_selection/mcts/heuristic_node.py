import numpy as np
from move_selection.mcts.abstract_node import AbstractNode


class HeuristicNode(AbstractNode):
    def __init__(self, position, parent, GameClass, network, c=np.sqrt(2), d=1, network_call_results=None,
                 verbose=False):
        super().__init__(position, parent, GameClass, c, verbose)
        self.network = network
        self.d = d

        if self.fully_expanded:
            self.heuristic = GameClass.get_winner(position)
            self.policy = None
            self.expansions = np.inf
        else:
            self.policy, self.heuristic = network.call(position[np.newaxis, ...])[0] if network_call_results is None \
                else network_call_results
            self.expansions = 0

    def count_expansions(self):
        return self.expansions

    def get_evaluation(self):
        return self.heuristic

    def expand(self, moves=None, network_call_results=None):
        if self.children is not None:
            raise Exception('Node already has children!')
        if self.fully_expanded:
            raise Exception('Node is terminal!')

        self.ensure_children(moves, network_call_results)
        if self.children is None:
            raise Exception('Failed to create children!')

        critical_value = max([child.heuristic for child in self.children]) if self.is_maximizing else \
            min([child.heuristic for child in self.children])
        self.heuristic = critical_value

        # update heuristic for all parents if it beats their current best heuristic
        node = self.parent
        while node is not None:
            if (node.is_maximizing and critical_value > node.heuristic) or \
                    (not node.is_maximizing and critical_value < node.heuristic):
                node.heuristic = critical_value
                node.expansions += 1
                node = node.parent
            else:
                node.expansions += 1
                node = node.parent
                # once a parent is reached that is not affected by the critical value,
                # all further parents are also not affected
                break

        # despite not updating heuristic values for further parents, continue to update their expansion counts
        while node is not None:
            node.expansions += 1
            node = node.parent

    def set_fully_expanded(self, minimax_evaluation):
        self.heuristic = minimax_evaluation
        self.expansions = np.inf
        self.fully_expanded = True

    def get_puct_heuristic_for_child(self, i):
        exploration_term = self.c * np.sqrt(np.log(self.expansions) / (self.children[i].expansions + 1))
        policy_term = self.d * self.policy[i]
        return exploration_term + policy_term

    def ensure_children(self, moves=None, network_call_results=None):
        if self.children is None:
            moves = self.GameClass.get_possible_moves(self.position) if moves is None else moves
            network_call_results = self.network.call(np.stack(moves, axis=0)) if network_call_results is None \
                else network_call_results
            self.children = [HeuristicNode(move, self, self.GameClass, self.network, self.c, self.d,
                                           network_call_results=network_call_result, verbose=self.verbose)
                             for move, network_call_result in zip(moves, network_call_results)]
            self.expansions = 1
