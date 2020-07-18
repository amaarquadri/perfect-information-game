import pickle
from src.ui.pygame_ui import PygameUI
from src.games.connect4 import Connect4 as GameClass


def view_game(path):
    with open(path, 'rb') as fin:
        training_data, result = pickle.load(fin)
    print('Result: ', result)

    positions = [position for position, _ in training_data]
    distributions = [f'Distribution: {distribution}' for _, distribution in training_data]

    ui = PygameUI(GameClass)
    ui.show_game(positions, messages=distributions)


if __name__ == '__main__':
    view_game(f'../heuristics/{GameClass.__name__}/games/rolling_mcts_network_games/game1593991058.990748.pickle')
