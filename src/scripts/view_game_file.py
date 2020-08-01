import pickle
from ..ui.pygame_ui import PygameUI
from ..utils.active_game import ActiveGame as GameClass
from ..utils.utils import get_training_path


def view_game(path):
    with open(path, 'rb') as fin:
        training_data, result = pickle.load(fin)
    print('Result: ', result)

    positions = [position for position, _ in training_data]
    distributions = [f'Distribution: {distribution}' for _, distribution in training_data]

    ui = PygameUI(GameClass)
    ui.show_game(positions, messages=distributions)


if __name__ == '__main__':
    view_game(f'{get_training_path(GameClass)}/games/reinforcement_learning_games/sample_game.pickle')
    view_game(f'{get_training_path(GameClass)}/games/rollout_mcts_games/sample_game.pickle')
