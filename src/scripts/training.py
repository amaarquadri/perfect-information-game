from utils.active_game import ActiveGame as GameClass
from learning.utils import train_from_config


if __name__ == '__main__':
    train_from_config(GameClass, 'training_config')
