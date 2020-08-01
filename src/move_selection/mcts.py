from abc import ABC, abstractmethod
from time import time
from multiprocessing import Process, Pipe, Pool
import numpy as np


class AsyncMCTS:
    """
    Implementation of Monte Carlo Tree Search that uses the other player's time to continue thinking.
    This is achieved using multiprocessing, and a Pipe for transferring data to and from the worker process.
    """

    def __init__(self, GameClass, position, time_limit=3, network=None, c=np.sqrt(2), d=1, threads=1):
        """
        Either:
        If network is provided, threads must be 1.
        If network is not provided, then threads will be used for leaf parallelization
        """
        if network is not None and threads != 1:
            raise Exception('Threads != 1 with Network != None')

        self.parent_pipe, worker_pipe = Pipe()
        self.worker_process = Process(target=self.loop_func,
                                      args=(GameClass, position, time_limit, network, c, d, threads, worker_pipe))

    def start(self):
        self.worker_process.start()

    def choose_move(self, user_chosen_position):
        """
        Instructs the worker thread that the user has chosen the move specified by the given position.
        The worker thread will then continue thinking for time_limit, and then return its chosen move.

        :param user_chosen_position: The board position resulting from the user's move.
        :return: The move chosen by monte carlo tree search.
        """
        self.parent_pipe.send(user_chosen_position)
        return self.parent_pipe.recv()

    def terminate(self):
        self.worker_process.terminate()
        self.worker_process.join()

    @staticmethod
    def loop_func(GameClass, position, time_limit, network, c, d, threads, worker_pipe):
        if network is None:
            pool = Pool(threads) if threads > 1 else None
            root = RolloutNode(position, parent=None, GameClass=GameClass, c=c, rollout_batch_size=threads, pool=pool,
                               verbose=True)
        else:
            network.initialize()
            root = HeuristicNode(position, None, GameClass, network, c, d, verbose=True)

        while True:
            # TODO: modify so that moves can be made by the user and ai in arbitrary order
            best_node = root.choose_expansion_node()

            if best_node is not None:
                best_node.expand()

            if root.children is not None and worker_pipe.poll():
                user_chosen_position = worker_pipe.recv()

                for child in root.children:
                    if np.all(child.position == user_chosen_position):
                        root = child
                        root.parent = None
                        break
                else:
                    print(user_chosen_position)
                    raise Exception('Invalid user chosen move!')

                if GameClass.is_over(root.position):
                    print('Game Over in Async MCTS: ', GameClass.get_winner(root.position))
                    break

                AsyncMCTS.manage_time(root, time_limit)

                print(f'MCTS choosing move based on {root.count_expansions()} expansions!')
                root = root.choose_best_node(optimal=True)
                print('Expected outcome: ', root.get_evaluation())
                root.parent = None
                worker_pipe.send(root.position)
                if GameClass.is_over(root.position):
                    print('Game Over in Async MCTS: ', GameClass.get_winner(root.position))
                    break

    @staticmethod
    def manage_time(root, time_remaining, p_min=0.05, p_max=0.7, p_step=0.01, f_second=0.9):
        """
        :param root:
        :param time_remaining: The total amount of time remaining on the AI's clock.
        :param p_min: The minimum fraction of the remaining time that should be used for this move.
        :param p_max: The maximum fraction of the remaining time that should be used for this move.
        :param p_step:
        :param f_second: An upper bound estimate of the fraction of expansions that will occur
                         at the second most visited child of the root node.
        """
        start_time = time()
        time_elapsed = 0
        expansions = 0

        for i, t_fraction in enumerate(np.concatenate(([p_min], np.linspace(p_min + p_step, p_max,
                                                                            int(np.rint((p_max - p_min) / p_step)),
                                                                            endpoint=True)))):
            if i != 0:
                visit_counts = sorted([child.count_expansions() for child in root.children])
                if len(visit_counts) < 2:
                    break
                visits_to_surpass = visit_counts[-1] - visit_counts[-2]
                if visits_to_surpass > f_second * expansions / time_elapsed * (p_max * time_remaining - time_elapsed):
                    break

            while time_elapsed < t_fraction * time_remaining:
                best_node = root.choose_expansion_node()

                # best_node will be None if the tree is fully expanded
                if best_node is None:
                    return

                best_node.expand()
                expansions += 1
                time_elapsed = time() - start_time


class MCTS:
    """
    Implementation of Monte Carlo Tree Search
    https://www.youtube.com/watch?v=UXW2yZndl7U
    """

    def __init__(self, GameClass, network=None, c=np.sqrt(2), d=1, threads=1):
        """
        Either:
        If network is provided, threads must be 1.
        If network is not provided, then threads will be used for leaf parallelization
        """
        if network is not None and threads != 1:
            raise Exception('Threads != 1 with Network != None')

        self.GameClass = GameClass
        self.network = network
        if network is not None:
            network.initialize()
        self.c = c
        self.d = d
        self.threads = threads
        self.pool = Pool(threads) if threads > 1 else None

    def choose_move(self, position, time_limit=10):
        if self.GameClass.is_over(position):
            raise Exception('Game Finished!')

        if self.network is None:
            root = RolloutNode(position, parent=None, GameClass=self.GameClass, c=self.c,
                               rollout_batch_size=self.threads, pool=self.pool, verbose=True)
        else:
            root = HeuristicNode(position, None, self.GameClass, self.network, self.c, self.d, verbose=True)

        start_time = time()
        while time() - start_time < time_limit:
            best_node = root.choose_expansion_node()

            # best_node will be None if the tree is fully expanded
            if best_node is None:
                break

            best_node.expand()

        best_child = root.choose_best_node(optimal=True)
        return best_child.position


class AbstractNode(ABC):
    def __init__(self, position, parent, GameClass, c=np.sqrt(2), verbose=False):
        self.position = position
        self.parent = parent
        self.GameClass = GameClass
        self.c = c
        self.fully_expanded = GameClass.is_over(position)
        self.is_maximizing = GameClass.is_player_1_turn(position)
        self.children = None
        self.verbose = verbose

    @abstractmethod
    def get_evaluation(self):
        pass

    @abstractmethod
    def count_expansions(self):
        pass

    @abstractmethod
    def ensure_children(self):
        pass

    @abstractmethod
    def get_puct_heuristic_for_child(self, i):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def set_fully_expanded(self, minimax_evaluation):
        pass

    def choose_best_node(self, return_probability_distribution=False, optimal=False):
        distribution = []

        optimal_value = 1 if self.is_maximizing else -1
        if self.fully_expanded:
            if self.verbose:
                if self.get_evaluation() == optimal_value:
                    print('I\'m going to win')
                elif self.get_evaluation() == 0:
                    print('It\'s a draw')
                else:
                    print('I resign')

            for child in self.children:
                # only consider children that result in the optimal outcome
                if child.fully_expanded and child.get_evaluation() == self.get_evaluation():
                    # TODO: when losing, consider the number of ways the opponent can win in response to a move
                    depth_to_endgame = child.depth_to_end_game()
                    # if we are winning, weight smaller depths much more strongly by using e^-x
                    # if we are losing or drawing, weight larger depths much more strongly by using e^x
                    relative_probability = np.exp(-depth_to_endgame if self.get_evaluation() == optimal_value else
                                                  depth_to_endgame)
                    distribution.append(relative_probability)
                else:
                    distribution.append(0)
        else:
            for child in self.children:
                if not child.fully_expanded:
                    distribution.append(child.count_expansions())
                elif child.get_evaluation() == -optimal_value:
                    # this move is guaranteed to lose
                    distribution.append(0)
                else:
                    # use the self.heuristic as a proxy for the chance of winning the game
                    # the greater the perceived chance of winning the less appealing a draw is, and vice versa
                    winning_chance = (self.get_evaluation() * optimal_value) / 2 + 0.5
                    distribution.append(self.count_expansions() * (1 - winning_chance))

        distribution = np.array(distribution) / sum(distribution)
        idx = np.argmax(distribution) if optimal else np.random.choice(np.arange(len(distribution)), p=distribution)
        best_child = self.children[idx]
        return (best_child, distribution) if return_probability_distribution else best_child

    def choose_expansion_node(self):
        # TODO: continue tree search in case the user makes a mistake and the game continues
        if self.fully_expanded:
            return None

        if self.count_expansions() == 0:
            return self

        self.ensure_children()
        best_heuristic = -np.inf if self.is_maximizing else np.inf
        best_child = None
        for i, child in enumerate(self.children):
            # don't bother exploring fully expanded children
            if child.fully_expanded:
                optimal_value = 1 if self.is_maximizing else -1
                # If this child is already optimal, then self is fully expanded and there is no point searching further
                if child.get_evaluation() == optimal_value:
                    self.set_fully_expanded(optimal_value)
                    return self.parent.choose_expansion_node() if self.parent is not None else None
                # continue searching other children, there may be another child that is more optimal
                continue

            # check puct heuristic before calling child.get_evaluation() because it may result in division by 0
            puct_heuristic = self.get_puct_heuristic_for_child(i)
            if np.isinf(puct_heuristic):
                return child

            if self.is_maximizing:
                combined_heuristic = child.get_evaluation() + puct_heuristic
                if combined_heuristic > best_heuristic:
                    best_heuristic = combined_heuristic
                    best_child = child
            else:
                combined_heuristic = child.get_evaluation() - puct_heuristic
                if combined_heuristic < best_heuristic:
                    best_heuristic = combined_heuristic
                    best_child = child

        # if nothing was found because all children are fully expanded
        if best_child is None:
            if self.verbose and not self.fully_expanded and self.parent is None:
                print('Fully expanded tree!')

            minimax_evaluation = max([child.get_evaluation() for child in self.children]) if self.is_maximizing \
                else min([child.get_evaluation() for child in self.children])
            self.set_fully_expanded(minimax_evaluation)
            # this node is now fully expanded, so ask the parent to try to choose again
            # if no parent is available (i.e. this is the root node) then the entire search tree has been expanded
            return self.parent.choose_expansion_node() if self.parent is not None else None

        return best_child.choose_expansion_node()

    def depth_to_end_game(self):
        if not self.fully_expanded:
            raise Exception('Node not fully expanded!')

        if self.children is None:
            return 0

        optimal_value = 1 if self.is_maximizing else -1
        if self.get_evaluation() == optimal_value:
            # if we are winning, win as fast as possible
            return 1 + min(child.depth_to_end_game() for child in self.children
                           if child.fully_expanded and child.get_evaluation() == self.get_evaluation())
        else:
            # if we are losing or it is a draw, lose as slow as possible
            return 1 + max(child.depth_to_end_game() for child in self.children
                           if child.fully_expanded and child.get_evaluation() == self.get_evaluation())


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
