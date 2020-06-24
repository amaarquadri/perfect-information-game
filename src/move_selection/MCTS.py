import numpy as np
from multiprocessing import Pool
from time import time


class MCTS:
    """
    Implementation of Monte Carlo Tree Search
    https://www.youtube.com/watch?v=UXW2yZndl7U
    """
    def __init__(self, GameClass, c=np.sqrt(2), threads=7):
        self.GameClass = GameClass
        self.c = c
        self.threads = threads
        self.pool = Pool(threads) if threads > 1 else None

    def choose_move(self, position, time_limit=10):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        root = Node(position, parent=None, GameClass=self.GameClass)
        start_time = time()
        while time() - start_time < time_limit:
            root.choose_rollout_node(self.c).rollout(self.threads, self.pool)

        best_heuristic = -np.inf if root.is_maximizing else np.inf
        best_child = None
        for child in root.children:
            if child.rollout_count > 0:
                child_heuristic = child.rollout_sum / child.rollout_count
                if (root.is_maximizing and child_heuristic > best_heuristic) or \
                        (not root.is_maximizing and child_heuristic < best_heuristic):
                    best_heuristic = child_heuristic
                    best_child = child
        # TODO: contemplate best_child while user is making their move
        return best_child.position


class Node:
    def __init__(self, position, parent, GameClass):
        self.position = position
        self.parent = parent
        self.GameClass = GameClass
        self.is_maximizing = GameClass.is_player_1_turn(position)

        self.children = None
        self.rollout_sum = 0
        self.rollout_count = 0

    def ensure_children(self):
        if self.children is None:
            self.children = [Node(move, self, self.GameClass) for move in
                             self.GameClass.get_possible_moves(self.position)]

    def choose_rollout_node(self, c=np.sqrt(2)):
        if self.rollout_count == 0:
            return self
        
        self.ensure_children()
        best_heuristic = -np.inf if self.is_maximizing else np.inf
        best_child = None
        for child in self.children:
            # any children that have not been visited will have an exploration term of infinity, so we can just choose
            # the first such child node
            if child.rollout_count == 0:
                return child

            child_heuristic = child.rollout_sum / child.rollout_count
            exploration_term = c * np.sqrt(np.log(self.rollout_count) / child.rollout_count)
            if self.is_maximizing:
                child_heuristic += exploration_term
                if child_heuristic > best_heuristic:
                    best_heuristic = child_heuristic
                    best_child = child
            else:
                child_heuristic -= exploration_term
                if child_heuristic < best_heuristic:
                    best_heuristic = child_heuristic
                    best_child = child

        return best_child.choose_rollout_node(c)

    def rollout(self, rollouts=7, pool=None):
        rollout_sum = sum(pool.starmap(self.execute_single_rollout, [() for _ in range(rollouts)]) if pool is not None
                          else [self.execute_single_rollout() for _ in range(rollouts)])

        # update this node and all its parents
        node = self
        while node is not None:
            node.rollout_sum += rollout_sum
            node.rollout_count += rollouts
            node = node.parent

    def execute_single_rollout(self):
        state = self.position
        while not self.GameClass.is_over(state):
            sub_states = self.GameClass.get_possible_moves(state)
            state = sub_states[np.random.randint(len(sub_states))]

        return self.GameClass.get_winner(state)
