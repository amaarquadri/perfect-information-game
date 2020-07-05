import numpy as np
from time import time
import pickle
from multiprocessing import Process, Pipe
from src.games.Connect4 import Connect4
from src.ui.pygame_ui import PygameUI
from src.move_selection.MCTS import RolloutNode, HeuristicNode
from src.heuristics.Network import Network


def simulate_game_batches_worker_process(GameClass, response_pipe, network, path='../heuristics/games/raw_mcts_games',
                                         expansions_per_move=800, game_batch_count=3, c=np.sqrt(2), d=1):
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
                print('Making move')
                best_node, distribution = root.choose_best_node(return_probability_distribution=True)
                training_data_sets[i].append((root.position, distribution))
                root = best_node
                root.parent = None
                if GameClass.is_over(root.position):
                    print('Saving game')
                    game = (training_data_sets[i], GameClass.get_winner(root.position))
                    with open(f'{path}/game{time()}.pickle', 'wb') as fout:
                        pickle.dump(game, fout)
                    response_pipe.send(game)

                    training_data_sets[i] = []
                    root = HeuristicNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, network=None,
                                         c=c, d=d, network_call_results=(np.copy(starting_policy), starting_evaluation))

                roots[i] = root

            best_node = root.choose_expansion_node()
            if best_node is None:
                while root.children is not None:
                    best_node, distribution = root.choose_best_node(return_probability_distribution=True)
                    training_data_sets[i].append((root.position, distribution))
                    root = best_node
                    root.parent = None
                game = (training_data_sets[i], GameClass.get_winner(root.position))
                with open(f'{path}/game{time()}.pickle', 'wb') as fout:
                    pickle.dump(game, fout)
                response_pipe.send(game)
                training_data_sets[i] = []
                root = HeuristicNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, network=None,
                                     c=c, d=d, network_call_results=(np.copy(starting_policy), starting_evaluation))
                roots[i] = root
                best_node = root.choose_expansion_node()
            best_nodes.append(best_node)

        best_nodes_moves = [GameClass.get_possible_moves(best_node.position) for best_node in best_nodes]
        network_call_results_batch = network.call(np.stack([position for moves in best_nodes_moves
                                                            for position in moves], axis=0))
        pos = 0
        for best_node, moves in zip(best_nodes, best_nodes_moves):
            new_pos = pos + len(moves)
            network_call_results = network_call_results_batch[pos:new_pos]
            best_node.expand(moves, network_call_results)
            pos = new_pos


def simulate_games_worker_process(GameClass, response_pipe, path='../heuristics/games/raw_mcts_games',
                                  expansions_per_move=800, network=None, c=np.sqrt(2), d=1):
    # format for each game file: ([(position, [pi_0, pi_1, ...]), ...], result)
    if network is not None:
        network.initialize()

    while True:
        training_data = []

        if network is None:
            root = RolloutNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass, c=c)
        else:
            root = HeuristicNode(GameClass.STARTING_STATE, None, GameClass, network, c, d)

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
        with open(f'{path}/game{time()}.pickle', 'wb') as fout:
            pickle.dump(game, fout)
        response_pipe.send(game)


def create_training_games(GameClass, num_games=10_000, threads=14, network_iteration=0,
                          model_path=None, c=np.sqrt(2), d=1):
    parent_training_data_pipe, worker_training_data_pipe = Pipe(duplex=False)
    if model_path is None:
        path = '../heuristics/games/raw_mcts_games'
        worker_processes = [Process(target=simulate_games_worker_process,
                                    args=(GameClass, worker_training_data_pipe, path, 800))
                            for _ in range(threads)]
        network_process = None
    else:
        path = f'../heuristics/games/mcts_network{network_iteration}_games'
        network_process, network_proxies = Network.spawn_process(GameClass, model_path, threads)
        worker_processes = [Process(target=simulate_games_worker_process,
                                    args=(GameClass, worker_training_data_pipe, path, 800, network_proxy, c, d))
                            for network_proxy in network_proxies]
        network_process.start()

    for worker_process in worker_processes:
        worker_process.start()

    games = [parent_training_data_pipe.recv() for _ in range(num_games)]

    for worker_process in worker_processes:
        worker_process.terminate()
    for worker_process in worker_processes:
        worker_process.join()
    if network_process is not None:
        network_process.terminate()
        network_process.join()
    return games


def create_dual_architecture_training_games(GameClass, num_games=10_000, threads_per_section=14,
                                            game_batch_count=4, network_iteration=0, c=np.sqrt(2), d=1):
    path = f'../heuristics/games/mcts_network{network_iteration}_games'
    model_path = f'../heuristics/models/model{network_iteration}.h5'
    network_process, network_a_proxies, network_b_proxies = Network.spawn_dual_architecture_process(GameClass,
                                                                                                    model_path,
                                                                                                    threads_per_section)
    parent_training_data_pipe, worker_training_data_pipe = Pipe(duplex=False)
    worker_a_processes = [Process(target=simulate_game_batches_worker_process,
                                  args=(GameClass, worker_training_data_pipe, network_proxy, path, 800,
                                        game_batch_count, c, d))
                          for network_proxy in network_a_proxies]
    worker_b_processes = [Process(target=simulate_game_batches_worker_process,
                                  args=(GameClass, worker_training_data_pipe, network_proxy, path, 800,
                                        game_batch_count, c, d))
                          for network_proxy in network_b_proxies]

    # start all processes in the order that they are required
    for worker_process in worker_a_processes:
        worker_process.start()
    network_process.start()
    for worker_process in worker_b_processes:
        worker_process.start()

    games = [parent_training_data_pipe.recv() for _ in range(num_games)]

    # terminate and join all processes
    for worker_process in worker_a_processes + worker_b_processes:
        worker_process.terminate()
    for worker_process in worker_a_processes + worker_b_processes:
        worker_process.join()
    network_process.terminate()
    network_process.join()

    return games


def train_network(GameClass=Connect4, iterations=8, threads=14):
    import os
    for iteration in range(iterations):
        create_training_games(GameClass, threads=threads, network_iteration=iteration,
                              model_path=f'../heuristics/models/model{iteration}.h5')

        data = []
        for file in os.listdir('../heuristics/games/raw_mcts_games'):
            with open(f'../heuristics/games/raw_mcts_games/{file}', 'rb') as fin:
                data.append(pickle.load(fin))

        network = Network(GameClass, f'../heuristics/models/model{iteration}.h5')
        network.train(data)
        network.save(f'../heuristics/models/model{iteration + 1}.h5')


def view_game(path):
    with open(path, 'rb') as fin:
        training_data, result = pickle.load(fin)
    print('Result: ', result)
    ui = PygameUI(Connect4)
    i = 0
    while True:
        val = ui.click_left_or_right()
        if val is None:
            return
        if val:
            i = min(i + 1, len(training_data) - 1)
        else:
            i = max(i - 1, 0)

        position, distribution = training_data[i]
        ui.draw(position)
        print(distribution)


if __name__ == '__main__':
    create_dual_architecture_training_games(Connect4, network_iteration=1,
                                            threads_per_section=14, game_batch_count=50, num_games=10_000)
    # train_network(Connect4, 1, threads=14)
    # view_game('../heuristics/games/raw_mcts_game/game.pickle')
