import os


def start_tensor_board():
    os.system('cmd /k "tensorboard --logdir ../heuristics/logs"')


if __name__ == '__main__':
    start_tensor_board()
