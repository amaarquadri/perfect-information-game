from multiprocessing import Pipe
from multiprocessing.context import Process
from time import time
import numpy as np
from perfect_information_game.move_selection import MoveChooser
from perfect_information_game.move_selection.mcts import TablebaseNode, RolloutNode, HeuristicNode
from perfect_information_game.tablebases import EmptyTablebaseManager
from perfect_information_game.utils import OptionalPool


class AsyncMCTS(MoveChooser):
    """
    Implementation of Monte Carlo Tree Search that uses the other player's time to continue thinking.
    This is achieved using multiprocessing, and a Pipe for transferring data to and from the worker process.
    """

    def __init__(self, GameClass, starting_position, time_limit=3, network=None, c=np.sqrt(2), d=1, threads=1,
                 tablebase_manager=None):
        """
        Either:
        If network is provided, threads must be 1.
        If network is not provided, then threads will be used for leaf parallelization
        """
        super().__init__(GameClass, starting_position)
        if network is not None and threads != 1:
            raise ValueError('Threads != 1 with Network != None')

        if tablebase_manager is None:
            tablebase_manager = EmptyTablebaseManager(GameClass)
        self.parent_pipe, worker_pipe = Pipe()
        self.worker_process = Process(target=self.loop_func,
                                      args=(GameClass, starting_position, time_limit, network, c, d, threads,
                                            tablebase_manager, worker_pipe))

    def start(self):
        self.worker_process.start()

    def report_user_move(self, user_chosen_move):
        """
        Reports the given user chosen move to the worker thread.
        This allows the search tree to be narrowed.

        :param user_chosen_move:
        """
        self.parent_pipe.send(user_chosen_move)
        self.position = user_chosen_move

    def choose_move(self, return_distribution=False):
        """
        Instructs the worker thread to decide on an optimal move.
        The worker thread will then continue thinking for time_limit, and then return a list of its chosen moves.
        If multiple states are passed through before the ai's turn is completed,
        then they will be the contents of the list. Otherwise the list will have a single state.

        :return: The moves chosen by monte carlo tree search.
        """
        self.parent_pipe.send(None)
        chosen_positions = self.parent_pipe.recv()
        self.position = chosen_positions[-1][0]
        return chosen_positions if return_distribution else [position for position, _ in chosen_positions]

    def terminate(self):
        self.worker_process.terminate()
        self.worker_process.join()

    @staticmethod
    def loop_func(GameClass, position, time_limit, network, c, d, threads, tablebase_manager, worker_pipe):
        with OptionalPool(threads) as pool:
            if network is not None:
                network.initialize()

            root = TablebaseNode.attempt_create(position, None, GameClass, tablebase_manager, verbose=True,
                                                backup_factory=lambda:
                                                RolloutNode(position, None, GameClass, c, threads, pool, verbose=True)
                                                if network is None else
                                                HeuristicNode(position, None, GameClass, network, c, d, verbose=True))

            while True:
                best_node = root.choose_expansion_node()

                if best_node is not None:
                    best_node.expand()

                if root.children is not None and worker_pipe.poll():
                    user_chosen_position = worker_pipe.recv()

                    if user_chosen_position is not None:
                        # an updated position has been received so we can truncate the tree
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
                    else:
                        # this move chooser has been requested to decide on a move via the choose_move function
                        start_time = time()
                        while time() - start_time < time_limit:
                            best_node = root.choose_expansion_node()

                            # best_node will be None if the tree is fully expanded
                            if best_node is None:
                                break

                            best_node.expand()

                        is_ai_player_1 = GameClass.is_player_1_turn(root.position)
                        chosen_positions = []
                        print(f'MCTS choosing move based on {root.count_expansions()} expansions!')

                        # choose moves as long as it is still the ai's turn
                        while GameClass.is_player_1_turn(root.position) == is_ai_player_1:
                            if root.children is None:
                                best_node = root.choose_expansion_node()
                                if best_node is not None:
                                    best_node.expand()
                            root, distribution = root.choose_best_node(return_probability_distribution=True, optimal=True)
                            chosen_positions.append((root.position, distribution))

                        print('Expected outcome: ', root.get_evaluation())
                        root.parent = None  # delete references to the parent and siblings
                        worker_pipe.send(chosen_positions)
                        if GameClass.is_over(root.position):
                            print('Game Over in Async MCTS: ', GameClass.get_winner(root.position))
                            break

    def reset(self):
        raise NotImplementedError('')
