import os
from src.games.Connect4 import Connect4


def start_tensor_board(GameClass, enable_venv_manually=True):
    venv_command = 'cd ../../venv/Scripts && activate && cd ../../src/scripts'
    tensor_board_command = f'tensorboard --logdir ../heuristics/{GameClass.__name__}/logs'
    if enable_venv_manually:
        command = f'cmd /k "{venv_command} && {tensor_board_command}"'
    else:
        command = f'cmd /k "{tensor_board_command}"'
    os.system(command)


if __name__ == '__main__':
    start_tensor_board(Connect4)
