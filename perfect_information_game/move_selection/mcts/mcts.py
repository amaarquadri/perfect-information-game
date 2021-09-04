from time import time
from multiprocessing import Pool
import numpy as np
from perfect_information_game.move_selection import MoveChooser
from perfect_information_game.move_selection.mcts import TablebaseNode, RolloutNode, HeuristicNode
from perfect_information_game.tablebases import EmptyTablebaseManager


# TODO: add hash table to keep track of multiple move combinations that lead to the same position


class MCTS(MoveChooser):
    """
    Implementation of Monte Carlo Tree Search
    https://www.youtube.com/watch?v=UXW2yZndl7U
    """

    def __init__(self, GameClass, starting_position=None, network=None, c=np.sqrt(2), d=1, threads=1,
                 tablebase_manager=None):
        """
        Either:
        If network is provided, threads must be 1.
        If network is not provided, then threads will be used for leaf parallelization
        """
        super().__init__(GameClass, starting_position)
        if network is not None and threads != 1:
            raise Exception('Threads != 1 with Network != None')

        self.network = network
        if network is not None:
            network.initialize()
        self.c = c
        self.d = d
        self.threads = threads
        self.pool = Pool(threads) if threads > 1 else None
        self.tablebase_manager = tablebase_manager if tablebase_manager is not None \
            else EmptyTablebaseManager(GameClass)

    def choose_move(self, return_distribution=False, time_limit=10):
        if return_distribution:
            # TODO: implement
            print('Returning distributions not implemented!')
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        root = TablebaseNode.attempt_create(self.position, None, self.GameClass, self.tablebase_manager, verbose=True,
                                            backup_factory=lambda:
                                            RolloutNode(self.position, None, self.GameClass, self.c, self.threads,
                                                        self.pool, verbose=True)
                                            if self.network is None else
                                            HeuristicNode(self.position, None, self.GameClass, self.network, self.c,
                                                          self.d, verbose=True))

        start_time = time()
        while time() - start_time < time_limit:
            best_node = root.choose_expansion_node()

            # best_node will be None if the tree is fully expanded
            if best_node is None:
                break

            best_node.expand()

        is_ai_player_1 = self.GameClass.is_player_1_turn(root.position)
        chosen_positions = []
        print(f'MCTS choosing move based on {root.count_expansions()} expansions!')

        # choose moves as long as it is still the ai's turn
        while self.GameClass.is_player_1_turn(root.position) == is_ai_player_1:
            if root.children is None:
                best_node = root.choose_expansion_node()
                if best_node is not None:
                    best_node.expand()
            root = root.choose_best_node(optimal=True)
            chosen_positions.append(root.position)

        print('Expected outcome: ', root.get_evaluation())
        return chosen_positions
