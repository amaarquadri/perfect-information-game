from time import time
import numpy as np
from src.move_selection.MCTS import RolloutNode
from src.games.Connect4 import Connect4


def benchmark(GameClass, trials=5):
    times = []
    for _ in range(trials):
        start_time = time()
        root = RolloutNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass)
        while root.count_expansions() < 1000:
            root.choose_expansion_node().expand()
        times.append(time() - start_time)
    print(np.mean(times))
    print(np.std(times))


if __name__ == '__main__':
    benchmark(Connect4)
