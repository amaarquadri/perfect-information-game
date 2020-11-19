import os
import pickle
from time import time
from multiprocessing import Process, Queue, Event
import numpy as np
from src.move_selection.mcts.rollout_node import RolloutNode
from src.move_selection.mcts.heuristic_node import HeuristicNode
from src.heuristics.network import Network
from src.utils.utils import get_training_path


class SelfPlayReinforcementLearning:
    def __init__(self, GameClass, model_path, threads_per_section=14, game_batch_count=6, expansions_per_move=500,
                 c=np.sqrt(2), d=1, buffer_size=1000):
        """
        If network is None, then self play will be done using random MCTS rollouts and saved to
        {get_training_path(GameClass)}/games/reinforcement_learning_games/
        """
        path = f'{get_training_path(GameClass)}/games/reinforcement_learning_games'
        self.network_process, network_a_proxies, network_b_proxies, network_training_data_pipe = \
            Network.spawn_dual_architecture_process(GameClass, model_path, threads_per_section)

        # We must use a queue instead of a pipe because the replay buffer process will not
        # immediately receive all the games that it is sent.
        # If a pipe was used instead, once the buffer fills up,
        # the send calls from the workers will block leading to a deadlock.
        # Unlike pipes, Queue have an "infinite" buffer, so put calls will never block.
        worker_training_data_queue = Queue()
        self.worker_a_processes = [Process(target=SelfPlayReinforcementLearning.game_batch_simulation_worker,
                                           args=(GameClass, worker_training_data_queue, network_proxy, path,
                                                 expansions_per_move, game_batch_count, c, d))
                                   for network_proxy in network_a_proxies]
        self.worker_b_processes = [Process(target=SelfPlayReinforcementLearning.game_batch_simulation_worker,
                                           args=(GameClass, worker_training_data_queue, network_proxy, path,
                                                 expansions_per_move, game_batch_count, c, d))
                                   for network_proxy in network_b_proxies]

        self.replay_buffer_process = Process(target=SelfPlayReinforcementLearning.replay_buffer_process_loop,
                                             args=(GameClass, worker_training_data_queue, network_training_data_pipe,
                                                   path, buffer_size))

    def start(self):
        # start all processes in a logical order
        self.network_process.start()  # has the most work to do at startup (i.e. loading the model)
        for worker_process in self.worker_a_processes:
            worker_process.start()
        for worker_process in self.worker_b_processes:
            worker_process.start()

        # has a lot of work to do at startup (filling replay buffer),
        # but won't actually need to do anything with it until the first game finishes
        self.replay_buffer_process.start()

    def terminate(self, timeout=0):
        if timeout != 0:
            print('Warning! Ignoring timeout!')

        self.replay_buffer_process.terminate()
        self.network_process.terminate()
        for worker_process in self.worker_a_processes + self.worker_b_processes:
            worker_process.terminate()

        self.replay_buffer_process.join()
        self.network_process.join()
        for worker_process in self.worker_a_processes + self.worker_b_processes:
            worker_process.join()

    @staticmethod
    def game_batch_simulation_worker(GameClass, response_queue, network, path,
                                     expansions_per_move, game_batch_count, c, d):
        """
        Simulates several games in series, and aggregates and batches all their network call requests.
        """
        network.initialize()
        training_data_sets = [[] for _ in range(game_batch_count)]
        starting_policy, starting_evaluation = network.call(GameClass.STARTING_STATE[np.newaxis, ...])[0]
        roots = [HeuristicNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, network=None, c=c, d=d,
                               network_call_results=(np.copy(starting_policy), starting_evaluation))
                 for _ in range(game_batch_count)]

        while True:
            best_nodes = []
            for i in range(game_batch_count):
                root = roots[i]
                if root.count_expansions() >= expansions_per_move:
                    best_node, distribution = root.choose_best_node(return_probability_distribution=True)
                    training_data_sets[i].append((root.position, distribution))
                    root = best_node
                    root.parent = None

                    # practically speaking this will never happen
                    if GameClass.is_over(root.position):
                        game = (training_data_sets[i], GameClass.get_winner(root.position))
                        game_file = f'{path}/game_{time()}.pickle'
                        with open(game_file, 'wb') as fout:
                            pickle.dump(game, fout)
                        response_queue.put((game_file, len(training_data_sets[i])))

                        training_data_sets[i] = []
                        root = HeuristicNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, network=None,
                                             c=c, d=d,
                                             network_call_results=(np.copy(starting_policy), starting_evaluation))

                    roots[i] = root

                best_node = root.choose_expansion_node()
                if best_node is None:
                    while root.children is not None:
                        best_node, distribution = root.choose_best_node(return_probability_distribution=True)
                        training_data_sets[i].append((root.position, distribution))
                        root = best_node
                        root.parent = None
                    game = (training_data_sets[i], GameClass.get_winner(root.position))
                    game_file = f'{path}/game_{time()}.pickle'
                    with open(game_file, 'wb') as fout:
                        pickle.dump(game, fout)
                    response_queue.put((game_file, len(training_data_sets[i])))
                    training_data_sets[i] = []
                    root = HeuristicNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, network=None,
                                         c=c, d=d, network_call_results=(np.copy(starting_policy), starting_evaluation))
                    roots[i] = root
                    best_node = root.choose_expansion_node()
                best_nodes.append(best_node)

            # batch evaluations for all possible moves for the best_node in all game_batch_count games
            best_nodes_moves = [GameClass.get_possible_moves(best_node.position) for best_node in best_nodes]
            network_call_results_batch = network.call(np.stack([position for moves in best_nodes_moves
                                                                for position in moves], axis=0))

            # un-batch network call results, and tell each best_node to expand with its respective network call results
            pos = 0
            for best_node, moves in zip(best_nodes, best_nodes_moves):
                new_pos = pos + len(moves)
                network_call_results = network_call_results_batch[pos:new_pos]
                best_node.expand(moves, network_call_results)
                pos = new_pos

    @staticmethod
    def replay_buffer_process_loop(GameClass, training_game_queue, network_training_pipe, path, buffer_size):
        game_files, game_lengths = SelfPlayReinforcementLearning.load_games(GameClass, path, buffer_size)

        while True:
            new_game_file, new_game_length = training_game_queue.get()

            if len(game_lengths) >= buffer_size:
                game_files.pop(0)
                game_lengths.pop(0)

            game_files.append(new_game_file)
            game_lengths.append(new_game_length)

            # exponential probability distribution biases towards more recently played games
            # probability density function is a normalized rescaling of e^x from (0, 1) to (0, N)
            # https://www.desmos.com/calculator/aoqionfo8j
            total = sum(game_lengths)
            probability_distribution = np.exp(np.arange(total) / total)
            probability_distribution = probability_distribution / np.sum(probability_distribution)
            indices = np.random.choice(np.arange(total), 256, replace=False, p=probability_distribution)

            # read from files to collect moves and their outcomes into dataset
            data = []
            game_lengths_cum_sum = np.cumsum(game_lengths)
            for index in indices:
                game_index = np.searchsorted(index, game_lengths_cum_sum, side='right')
                move_index = index - game_lengths_cum_sum[game_index - 1]
                with open(game_files[game_index], 'rb') as fin:
                    training_data, outcome = pickle.load(fin)
                    state, policy = training_data[move_index]
                    data.append(([(state, policy)], outcome))

            states, (policies, values) = Network.process_data(GameClass, data)
            print('Starting Training step')
            network_training_pipe.send((states, policies, values))

    @staticmethod
    def load_games(GameClass, path, count):
        game_files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file[-7:] == '.pickle']
        if len(game_files) < count:
            backup_path = f'{get_training_path(GameClass)}/games/rollout_mcts_games'
            backup_game_files = [os.path.join(backup_path, file) for file in sorted(os.listdir(backup_path))
                                 if file[-7:] == '.pickle']
            game_files = backup_game_files + game_files
        game_files = game_files[-count:]

        if len(game_files) < 2:
            raise Exception('Not enough games for the replay buffer!')
        if len(game_files) < count:
            print(f'Warning! Not enough games to fill the replay buffer. Starting with {len(game_files)} games.')

        game_lengths = []
        for game_file in game_files:
            with open(game_file, 'rb') as fin:
                game = pickle.load(fin)
                training_data = game[0]
                game_lengths.append(len(training_data))

        return game_files, game_lengths


class MCTSRolloutGameGenerator:
    def __init__(self, GameClass, threads=14, expansions_per_move=1000, c=np.sqrt(2)):
        path = f'{get_training_path(GameClass)}/games/rollout_mcts_games'
        self.termination_event = Event()
        self.worker_processes = [Process(target=MCTSRolloutGameGenerator.simulate_games_worker_process,
                                         args=(GameClass, self.termination_event, path, expansions_per_move, c))
                                 for _ in range(threads)]

    def start(self):
        for worker_process in self.worker_processes:
            worker_process.start()

    def terminate(self, timeout=3600):
        # gently terminate, allowing each child process a specified amount of time to finish its current task
        self.termination_event.set()
        start_time = time()
        for worker_process in self.worker_processes:
            remaining_time = timeout - (time() - start_time)
            if remaining_time > 0:
                worker_process.join(remaining_time)
                if not worker_process.is_alive():
                    continue

            print('Force terminating worker')
            worker_process.terminate()
            worker_process.join()

    @staticmethod
    def simulate_games_worker_process(GameClass, termination_event, path, expansions_per_move, c):
        # format for each game file: ([(position, [pi_0, pi_1, ...]), ...], result)
        while not termination_event.is_set():
            training_data = []

            root = RolloutNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, c=c)

            while not GameClass.is_over(root.position):
                while root.count_expansions() < expansions_per_move:
                    best_node = root.choose_expansion_node()

                    # best_node will be None if the tree is fully expanded
                    if best_node is None:
                        break

                    best_node.expand()

                best_node, distribution = root.choose_best_node(return_probability_distribution=True)
                training_data.append((root.position, distribution))
                root = best_node
                root.parent = None

            game = (training_data, GameClass.get_winner(root.position))
            with open(f'{path}/game_{time()}.pickle', 'wb') as fout:
                pickle.dump(game, fout)
