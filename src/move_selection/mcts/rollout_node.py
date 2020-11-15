import numpy as np
from move_selection.mcts.abstract_node import AbstractNode


class RolloutNode(AbstractNode):
    def __init__(self, position, parent, GameClass, c=np.sqrt(2), rollout_batch_size=1, pool=None, verbose=False):
        super().__init__(position, parent, GameClass, c, verbose)
        self.rollout_batch_size = rollout_batch_size
        self.pool = pool

        if self.fully_expanded:
            self.rollout_sum = GameClass.get_winner(position)
            self.rollout_count = np.inf
        else:
            self.rollout_sum = 0
            self.rollout_count = 0

    def count_expansions(self):
        return self.rollout_count

    def get_evaluation(self):
        return self.rollout_sum / self.rollout_count if not self.fully_expanded else self.rollout_sum

    def ensure_children(self):
        if self.children is None:
            self.children = [RolloutNode(move, self, self.GameClass, self.c, self.rollout_batch_size, self.pool,
                                         self.verbose)
                             for move in self.GameClass.get_possible_moves(self.position)]

    def set_fully_expanded(self, minimax_evaluation):
        self.rollout_sum = minimax_evaluation
        self.rollout_count = np.inf
        self.fully_expanded = True

    def get_puct_heuristic_for_child(self, i):
        exploration_term = self.c * np.sqrt(np.log(self.rollout_count) / self.children[i].rollout_count) \
            if self.children[i].rollout_count > 0 else np.inf
        return exploration_term

    def expand(self):
        rollout_sum = sum(self.pool.starmap(self.execute_single_rollout, [() for _ in range(self.rollout_batch_size)])
                          if self.pool is not None else
                          [self.execute_single_rollout() for _ in range(self.rollout_batch_size)])

        # update this node and all its parents
        node = self
        while node is not None:
            node.rollout_sum += rollout_sum
            node.rollout_count += self.rollout_batch_size
            node = node.parent

    def execute_single_rollout(self):
        state = self.position
        while not self.GameClass.is_over(state):
            sub_states = self.GameClass.get_possible_moves(state)
            state = sub_states[np.random.randint(len(sub_states))]

        return self.GameClass.get_winner(state)
