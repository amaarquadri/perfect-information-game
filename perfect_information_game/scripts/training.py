from perfect_information_game.games import Chess as GameClass
from perfect_information_game.learning import train_from_config


if __name__ == '__main__':
    train_from_config(GameClass, 'training_config')
