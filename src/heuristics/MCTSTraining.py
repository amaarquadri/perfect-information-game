import numpy as np
from multiprocessing import Pool
from time import time
import pickle
from src.games.Connect4 import Connect4
from src.move_selection.MCTS import Node
from src.heuristics.Network import Network


def simulate_game(GameClass, time_limit=30, c=np.sqrt(2), threads=1, pool=None):
    training_data = []
    root = Node(GameClass(), None, GameClass)

    while not GameClass.is_over(root):
        start_time = time()
        while time() - start_time < time_limit:
            best_node = root.choose_rollout_node(c)

            # best_node will be None if the tree is fully expanded
            if best_node is None:
                break

            best_node.rollout(threads, pool)

        best_node, distribution = root.choose_best_node(return_probability_distribution=True)
        training_data.append((root.position, distribution))
        root = best_node
        root.parent = None
    return training_data, GameClass.get_winner(root.position)


def training(time_limit=10*3600, threads=7):
    data = []
    start_time = time()
    pool = Pool(7) if threads > 1 else None
    while time() - start_time < time_limit:
        data.append(simulate_game(Connect4, threads=threads, pool=pool))
    with open('data.pickle') as fout:
        pickle.dump(data, fout)
    net = Network(input_shape=Connect4.STATE_SHAPE)
    net.train(data)
    net.save('model.h5')