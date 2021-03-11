import os
from perfect_information_game.utils import get_training_path, ActiveGame as GameClass


def start_tensor_board(enable_venv_manually=False):
    tensor_board_command = f'tensorboard --logdir {get_training_path(GameClass)}/logs'
    if enable_venv_manually:
        venv_command = 'cd ../../venv/Scripts && activate && cd ../../perfect_information_game/scripts'
        command = f'cmd /k "{venv_command} && {tensor_board_command}"'
    else:
        command = f'cmd /k "{tensor_board_command}"'
    os.system(command)


if __name__ == '__main__':
    start_tensor_board()
