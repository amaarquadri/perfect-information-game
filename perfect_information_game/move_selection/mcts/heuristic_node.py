import numpy as np
from perfect_information_game.move_selection.mcts import AbstractNode, TablebaseNode


class HeuristicNode(AbstractNode):
    def __init__(self, position, parent, GameClass, network, c=np.sqrt(2), d=1, network_call_results=None,
                 tablebase_manager=None, verbose=False):
        super().__init__(position, parent, GameClass, tablebase_manager, verbose)
        self.network = network
        self.c = c
        self.d = d

        self.policy, self.heuristic = network.call(position[np.newaxis, ...])[0] if network_call_results is None \
            else network_call_results
        self.expansions = 0

    def count_expansions(self):
        return self.expansions

    def get_evaluation(self):
        return self.outcome if self.fully_expanded else self.heuristic

    def expand(self, moves=None, network_call_results=None):
        """
        When a HeuristicNode is expanded, its children are created and they're heuristics are created with the network.
        Then the tree is updated so that every node's heuristic is the minimax value of its children.
        """
        if self.children is not None:
            # this node was already expanded
            raise Exception('Node already has children!')
        if self.fully_expanded:
            raise Exception('Node is fully expanded!')

        self.ensure_children(moves, network_call_results)
        if self.children is None:
            # this check is needed to prevent lint warnings
            raise Exception('Failed to create children!')

        critical_value = (max if self.is_maximizing else min)(
            [child.get_evaluation() for child in self.children])
        self.heuristic = critical_value

        # update heuristic for all parents if it beats their current best heuristic
        node = self.parent
        while node is not None:
            if critical_value > node.heuristic if node.is_maximizing else critical_value < node.heuristic:
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
        super(HeuristicNode, self).set_fully_expanded(minimax_evaluation)
        self.expansions = np.inf

    def get_puct_heuristic_for_child(self, i):
        exploration_term = self.c * np.sqrt(np.log(self.expansions) / (self.children[i].expansions + 1))
        policy_term = self.d * self.policy[i]
        return exploration_term + policy_term

    def ensure_children(self, moves=None, network_call_results=None):
        if self.children is not None:
            return
        moves = self.GameClass.get_possible_moves(self.position) if moves is None else moves
        network_call_results = self.network.call(np.stack(moves, axis=0)) if network_call_results is None \
            else network_call_results
        self.children = [TablebaseNode.attempt_create(move, self, self.GameClass, self.tablebase_manager, self.verbose,
                                                      lambda: HeuristicNode(move, self, self.GameClass, self.network,
                                                                            self.c, self.d, network_call_result,
                                                                            self.verbose))
                         for move, network_call_result in zip(moves, network_call_results)]
        self.expansions = 1
