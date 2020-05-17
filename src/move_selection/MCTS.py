import numpy as np


class MCTS:
    """
    Implementation of Monte Carlo Tree Search
    https://www.youtube.com/watch?v=UXW2yZndl7U
    """
    def __init__(self, GameClass, c=np.sqrt(2)):
        self.GameClass = GameClass
        self.c = c

    def choose_move(self, position, iterations=1000):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        root = Node(position, parent=None, GameClass=self.GameClass)
        for _ in range(iterations):
            root.search_and_rollout(self.c)

        best_heuristic = -np.inf if root.is_maximizing else np.inf
        best_child = None
        for child in root.children:
            if child.rollout_count > 0:
                child_heuristic = child.rollout_sum / child.rollout_count
                if (root.is_maximizing and child_heuristic > best_heuristic) or \
                        (not root.is_maximizing and child_heuristic < best_heuristic):
                    best_heuristic = child_heuristic
                    best_child = child
        return best_child.position


class Node:
    def __init__(self, position, parent, GameClass):
        self.position = position
        self.parent = parent
        self.GameClass = GameClass
        self.is_maximizing = GameClass.is_player_1_turn(position)

        if GameClass.is_over(position):
            self.terminal = True
            self.expanded = True
            self.children = []
            self.rollout_sum = GameClass.get_winner(position)
            self.rollout_count = 1
        else:
            self.terminal = False
            self.expanded = False
            self.children = None
            self.rollout_sum = 0
            self.rollout_count = 0

    def ensure_children(self):
        if self.children is None:
            self.children = [Node(move, self, self.GameClass) for move in
                             self.GameClass.get_possible_moves(self.position)]

    def search_and_rollout(self, c=np.sqrt(2)):
        if not self.expanded:
            self.rollout()
        else:
            best_heuristic = -np.inf if self.is_maximizing else np.inf
            best_child = None
            for child in self.children:
                if child.rollout_count > 0:
                    child_heuristic = child.rollout_sum / child.rollout_count + \
                                      c * np.sqrt(np.log(self.rollout_count) / child.rollout_count)
                    if (self.is_maximizing and child_heuristic > best_heuristic) or \
                            (not self.is_maximizing and child_heuristic < best_heuristic):
                        best_heuristic = child_heuristic
                        best_child = child
            best_child.search_and_rollout(c)

    def rollout(self):
        if self.terminal:
            rollout_sum = self.rollout_sum
        else:
            self.ensure_children()
            rollout_sum = self.children[np.random.randint(len(self.children))].rollout_recursive()

        # update this node and all its parents
        node = self
        while node is not None:
            node.rollout_sum += rollout_sum
            node.rollout_count += 1
            node = node.parent

        # mark this node as expanded
        self.expanded = True

    def rollout_recursive(self):
        if self.terminal:
            return self.rollout_sum

        self.ensure_children()
        val = self.children[np.random.randint(len(self.children))].rollout_recursive()

        # Update this node while recursing back up to the original call in rollout
        self.rollout_sum += val
        self.rollout_count += 1
        return val
