import numpy as np
from time import time
import pickle
from multiprocessing import Process, Pipe
from src.games.Connect4 import Connect4
from src.ui.pygame_ui import PygameUI
from src.move_selection.MCTS import RolloutNode, HeuristicNode
from src.heuristics.Network import Network


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


def create_training_games(GameClass, num_games=1000, threads=1, network_iteration=0,
                          policy_path=None, evaluation_path=None, c=np.sqrt(2), d=1):
    parent_training_data_pipe, worker_training_data_pipe = Pipe(duplex=False)
    if policy_path is None:
        path = '../heuristics/games/raw_mcts_games'
        worker_processes = [Process(target=simulate_games_worker_process,
                                    args=(GameClass, worker_training_data_pipe, path, 500))
                            for _ in range(threads)]
        network_process = None
    else:
        path = f'../heuristics/games/mcts_network{network_iteration}_games'
        network_process, network_proxies = Network.spawn_process(GameClass, policy_path, evaluation_path, threads)
        worker_processes = [Process(target=simulate_games_worker_process,
                                    args=(GameClass, worker_training_data_pipe, path, 100, network_proxy, c, d))
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


def train_network(GameClass=Connect4, iterations=8, threads=14):
    for iteration in range(iterations):
        games = create_training_games(GameClass, threads=threads, network_iteration=iteration,
                                      policy_path=f'../heuristics/models/policy{iteration}.h5',
                                      evaluation_path=f'../heuristics/models/evaluation{iteration}.h5')

        network = Network(GameClass, f'../heuristics/models/policy{iteration}.h5',
                          f'../heuristics/models/evaluation{iteration}.h5')
        network.train(games)
        network.save(f'../heuristics/models/policy{iteration + 1}.h5',
                     f'../heuristics/models/evaluation{iteration + 1}.h5')


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
    train_network(Connect4, 8, threads=14)
    # view_game('../heuristics/games/raw_mcts_game/game.pickle')
