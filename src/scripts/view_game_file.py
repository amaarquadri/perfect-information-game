import pickle
from src.ui.pygame_ui import PygameUI
from src.utils.utils import ActiveGame as GameClass


def view_game(path):
    with open(path, 'rb') as fin:
        training_data, result = pickle.load(fin)
    print('Result: ', result)

    positions = [position for position, _ in training_data]
    distributions = [f'Distribution: {distribution}' for _, distribution in training_data]

    ui = PygameUI(GameClass)
    ui.show_game(positions, messages=distributions)


if __name__ == '__main__':
    view_game(f'../../training/{GameClass.__name__}/games/reinforcement_learning_games/game1595094819.10189.pickle')
    view_game(f'../../training/{GameClass.__name__}/games/rollout_mcts_games/game1593815499.7633.pickle')
