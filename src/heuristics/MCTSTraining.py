import numpy as np
from multiprocessing import Pool
from time import time
import pickle
from src.games.Connect4 import Connect4
from src.move_selection.MCTS import Node
from src.ui.pygame_ui import PygameUI


def simulate_games_worker_process(GameClass, rollouts_per_move=500, c=np.sqrt(2), time_limit=2 * 3600):
    start_time = time()
    while time() - start_time < time_limit:
        training_data = []
        root = Node(GameClass.STARTING_STATE, None, GameClass)

        while not GameClass.is_over(root.position):
            while root.rollout_count < rollouts_per_move:
                best_node = root.choose_rollout_node(c)

                # best_node will be None if the tree is fully expanded
                if best_node is None:
                    break

                best_node.rollout(rollouts=1, pool=None)

            best_node, distribution = root.choose_best_node(return_probability_distribution=True)
            training_data.append((root.position, distribution))
            root = best_node
            root.parent = None

        with open(f'mcts_games/game{time()}.pickle', 'wb') as fout:
            pickle.dump((training_data, GameClass.get_winner(root.position)), fout)


def training(threads=14):
    with Pool(threads) as pool:
        pool.map(simulate_games_worker_process, [Connect4 for _ in range(threads)])


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
    training()
    # view_game('mcts_games/game1593296639.1298165.pickle')
