import numpy as np
from perfect_information_game.move_selection.mcts import AbstractNode, TablebaseNode
from perfect_information_game.utils import OptionalPool, choose_random


class RolloutNode(AbstractNode):
    def __init__(self, position, parent, GameClass, c=np.sqrt(2), rollout_batch_size=1, pool=None,
                 tablebase_manager=None, verbose=False):
        super().__init__(position, parent, GameClass, tablebase_manager, verbose)
        self.c = c
        self.rollout_batch_size = rollout_batch_size
        self.pool = pool if pool is not None else OptionalPool(threads=1)

        # track the sum and the number of rollouts so that the average can be updated as more rollouts are done.
        self.rollout_sum = 0
        self.rollout_count = 0

    def count_expansions(self):
        return self.rollout_count if not self.fully_expanded else np.inf

    def get_evaluation(self):
        return self.rollout_sum / self.rollout_count if not self.fully_expanded else self.outcome

    def ensure_children(self):
        if self.children is not None:
            return
        self.children = [TablebaseNode.attempt_create(move, self, self.GameClass, self.tablebase_manager, self.verbose,
                                                      lambda: RolloutNode(move, self, self.GameClass, self.c,
                                                                          self.rollout_batch_size, self.pool,
                                                                          self.verbose))
                         for move in self.GameClass.get_possible_moves(self.position)]

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
