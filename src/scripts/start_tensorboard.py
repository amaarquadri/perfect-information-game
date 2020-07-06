import os


def start_tensor_board(enable_venv_manually=True):
    venv_command = 'cd ../../venv/Scripts && activate && cd ../../src/scripts'
    tensor_board_command = 'tensorboard --logdir ../heuristics/logs'
    if enable_venv_manually:
        command = f'cmd /k "{venv_command} && {tensor_board_command}"'
    else:
        command = f'cmd /k "{tensor_board_command}"'
    os.system(command)


if __name__ == '__main__':
    start_tensor_board()
