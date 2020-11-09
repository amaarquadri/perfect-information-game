import os
from src.utils.active_game import ActiveGame as GameClass
from src.utils.utils import get_training_path


def start_tensor_board(enable_venv_manually=False):
    tensor_board_command = f'tensorboard --logdir ../../training/Connect4/logs'
    if enable_venv_manually:
        venv_command = 'cd ../../venv/Scripts && activate && cd ../../src/scripts'
        command = f'cmd /k "{venv_command} && {tensor_board_command}"'
    else:
        command = f'cmd /k "{tensor_board_command}"'
    os.system(command)


if __name__ == '__main__':
    start_tensor_board()
