import numpy as np


class MCTS:
    """
    Implementation of Monte Carlo Tree Search
    https://www.youtube.com/watch?v=UXW2yZndl7U
    """
    def __init__(self, GameClass, c=np.sqrt(2)):
        self.GameClass = GameClass
        self.c = c

    def choose_move(self, position, iterations=100):
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
            self.children = []
            self.rollout_sum = GameClass.get_winner(position)
            self.rollout_count = 1
        else:
            self.terminal = False
            self.children = None
            self.rollout_sum = 0
            self.rollout_count = 0

    def ensure_children(self):
        if self.children is None:
            self.children = [Node(move, self, self.GameClass) for move in
                             self.GameClass.get_possible_moves(self.position)]

    def search_and_rollout(self, c=np.sqrt(2)):
        if self.rollout_count == 0:
            self.rollout()
            return
        
        self.ensure_children()
        best_heuristic = -np.inf if self.is_maximizing else np.inf
        best_child = None
        for child in self.children:
            if child.rollout_count == 0:
                best_child = child
                break

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
        best_child.search_and_rollout(c)

    def rollout(self):
        if self.terminal:
            raise Exception('Not Implemented!')

        self.ensure_children()
        state = self.children[np.random.randint(len(self.children))].position
        while not self.GameClass.is_over(state):
            sub_states = self.GameClass.get_possible_moves(state)
            state = sub_states[np.random.randint(len(sub_states))]

        value = self.GameClass.get_winner(state)

        # update this node and all its parents
        node = self
        while node is not None:
            node.rollout_sum += value
            node.rollout_count += 1
            node = node.parent
