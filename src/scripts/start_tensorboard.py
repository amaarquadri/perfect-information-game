import os
from src.utils.utils import ActiveGame as GameClass


def start_tensor_board(enable_venv_manually=False):
    tensor_board_command = f'tensorboard --logdir ../heuristics/{GameClass.__name__}/logs'
    if enable_venv_manually:
        venv_command = 'cd ../../venv/Scripts && activate && cd ../../src/scripts'
        command = f'cmd /k "{venv_command} && {tensor_board_command}"'
    else:
        command = f'cmd /k "{tensor_board_command}"'
    os.system(command)


if __name__ == '__main__':
    start_tensor_board()
