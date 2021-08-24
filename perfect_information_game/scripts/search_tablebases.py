from perfect_information_game.tablebases import ChessTablebaseManager
from perfect_information_game.utils import get_training_path
from perfect_information_game.games import Chess as GameClass
import pickle
import numpy as np


def search_for_draws():
    manager = ChessTablebaseManager(GameClass)
    for descriptor in manager.available_tablebases:
        with open(f'{get_training_path(GameClass)}/tablebases/{descriptor}.pickle', 'rb') as file:
            nodes = pickle.load(file)
            print(descriptor)
            print('1: ', sum([GameClass.parse_move_bytes(move_bytes)[1] == 1 for move_bytes in nodes.values()]))
            print('0: ', sum([GameClass.parse_move_bytes(move_bytes)[1] == 0 for move_bytes in nodes.values()]))
            print('0 (not immediate stalemate): ', [GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))
                                                    for board_bytes, move_bytes in nodes.items()
                                                    if GameClass.parse_move_bytes(move_bytes)[1] == 0 and
                                                    not GameClass.is_over(GameClass.parse_board_bytes(board_bytes))])
            print('-1: ', sum([GameClass.parse_move_bytes(data)[1] == -1 for data in nodes.values()]))


def search_for_puzzles(descriptor='KBNk'):
    manager = ChessTablebaseManager(GameClass)
    blacklist = ['K1k5/2N4B/8/8/8/8/8/8 w - - - -']

    with open(f'{get_training_path(GameClass)}/tablebases/{descriptor}.pickle', 'rb') as file:
        nodes = pickle.load(file)
        print(descriptor)
        longest_terminal_distance = -1
        best_board_bytes = None
        for board_bytes, move_bytes in nodes.items():
            _, outcome, terminal_distance = GameClass.parse_move_bytes(move_bytes)
            if outcome == 1 and terminal_distance > longest_terminal_distance \
                    and GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes)) not in blacklist:
                best_board_bytes = board_bytes
                longest_terminal_distance = terminal_distance
        print(GameClass.encode_fen(GameClass.parse_board_bytes(best_board_bytes)))


def search_for_draw_by_repetition():
    manager = ChessTablebaseManager(GameClass)
    for descriptor in manager.available_tablebases:
        with open(f'{get_training_path(GameClass)}/tablebases/{descriptor}.pickle', 'rb') as file:
            nodes = pickle.load(file)
            print(descriptor)
            print('0 (draw by repetition): ', [GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))
                                               for board_bytes, move_bytes in nodes.items()
                                               if GameClass.parse_move_bytes(move_bytes)[-1] == np.inf])


if __name__ == '__main__':
    search_for_puzzles()
