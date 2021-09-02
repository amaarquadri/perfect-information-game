import unittest
import urllib.request
import os
import numpy as np
from perfect_information_game.games import Chess as GameClass
from perfect_information_game.tablebases import ChessTablebaseManager
import chess
import chess.syzygy


class TestChessTablebase(unittest.TestCase):
    @staticmethod
    def check_tablebase_for_major_piece(manager, descriptor, mate_in):
        if descriptor not in manager.available_tablebases:
            return

        mate_in_plies = 2 * mate_in  # mate in n is n + (n - 1) plies or 2 * n plies
        manager.ensure_loaded(descriptor)
        tablebase = manager.tablebases[descriptor]
        for board_bytes, move_bytes in tablebase.items():
            _, outcome, terminal_distance = GameClass.parse_move_bytes(move_bytes)
            if terminal_distance < 0:
                raise AssertionError(f'{descriptor} endgame with negative terminal distance!')
            if outcome == 1:
                if terminal_distance > mate_in_plies:
                    raise AssertionError(f'{descriptor} endgame that takes more than {mate_in} moves to win!')
            elif outcome == 0:
                state = GameClass.parse_board_bytes(board_bytes)
                if GameClass.is_player_1_turn(state):
                    raise AssertionError(f'White to move in {descriptor} endgame is a draw!')
                if terminal_distance > 0 and not any([GameClass.get_position_descriptor(move) == 'Kk'
                                                      for move in GameClass.get_possible_moves(state)]):
                    raise AssertionError(f'Black to move in {descriptor} endgame where the major piece '
                                         'cannot be captured is a draw!')
            else:
                raise AssertionError(f'{descriptor} endgame is lost!')

    @staticmethod
    def check_longest_mate(manager, descriptor, longest_win, longest_loss=-1):
        if descriptor not in manager.available_tablebases:
            return

        longest_win_plies = 2 * longest_win  # mate in n is n + (n - 1) plies or 2 * n plies
        longest_loss_plies = 2 * longest_loss  # mate in n is n + n plies
        manager.ensure_loaded(descriptor)
        tablebase = manager.tablebases[descriptor]
        for board_bytes, move_bytes in tablebase.items():
            _, outcome, terminal_distance = GameClass.parse_move_bytes(move_bytes)
            if terminal_distance < 0:
                raise AssertionError(f'{descriptor} endgame with negative terminal distance!')
            if outcome == 1:
                if terminal_distance > longest_win_plies:
                    raise AssertionError(f'{descriptor} endgame that takes more than {longest_win} moves to win!')
            elif outcome == -1:
                if terminal_distance > longest_loss_plies:
                    raise AssertionError(f'{descriptor} endgame that takes more than {longest_win} moves to lose!')

    def test_manually(self):
        """
        Tests for
        http://kirill-kryukov.com/chess/longest-checkmates/longest-checkmates.shtml
        """
        manager = ChessTablebaseManager(GameClass)
        # TestTablebaseGenerator.check_tablebase_for_major_piece(manager, 'KQk', 10)
        # TestTablebaseGenerator.check_tablebase_for_major_piece(manager, 'KRk', 16)
        # TestTablebaseGenerator.check_longest_mate(manager, 'KPk', 28)

        # 4 man no enemy
        TestChessTablebase.check_longest_mate(manager, 'KPPk', 32)
        TestChessTablebase.check_longest_mate(manager, 'KNNk', 1)
        TestChessTablebase.check_longest_mate(manager, 'KNPk', 27)
        TestChessTablebase.check_longest_mate(manager, 'KBPk', 32)
        TestChessTablebase.check_longest_mate(manager, 'KBNk', 33)
        TestChessTablebase.check_longest_mate(manager, 'KBBk', 13)
        TestChessTablebase.check_longest_mate(manager, 'KRPk', 16)
        TestChessTablebase.check_longest_mate(manager, 'KRNk', 16)
        TestChessTablebase.check_longest_mate(manager, 'KRBk', 16)
        TestChessTablebase.check_longest_mate(manager, 'KRRk', 7)
        TestChessTablebase.check_longest_mate(manager, 'KQPk', 10)
        TestChessTablebase.check_longest_mate(manager, 'KQNk', 9)
        TestChessTablebase.check_longest_mate(manager, 'KQBk', 8)
        TestChessTablebase.check_longest_mate(manager, 'KQRk', 6)
        TestChessTablebase.check_longest_mate(manager, 'KQQk', 4)

        # 4 man with enemy
        TestChessTablebase.check_longest_mate(manager, 'KPkp', 33, 33)
        TestChessTablebase.check_longest_mate(manager, 'KNkp', 7, 29)
        TestChessTablebase.check_longest_mate(manager, 'KBkp', 1, 29)
        TestChessTablebase.check_longest_mate(manager, 'KRkp', 26, 43)
        TestChessTablebase.check_longest_mate(manager, 'KQkp', 28, 29)
        TestChessTablebase.check_longest_mate(manager, 'KNkn', 1, 1)
        TestChessTablebase.check_longest_mate(manager, 'KBkn', 1, 1)
        TestChessTablebase.check_longest_mate(manager, 'KRkn', 40, 1)
        TestChessTablebase.check_longest_mate(manager, 'KQkn', 21)
        TestChessTablebase.check_longest_mate(manager, 'KBkb', 1, 1)
        TestChessTablebase.check_longest_mate(manager, 'KRkb', 29)
        TestChessTablebase.check_longest_mate(manager, 'KQkb', 17)
        TestChessTablebase.check_longest_mate(manager, 'KRkr', 19, 19)
        TestChessTablebase.check_longest_mate(manager, 'KQkr', 35, 19)
        TestChessTablebase.check_longest_mate(manager, 'KQkq', 13, 13)

    def test_puzzles(self):
        manager = ChessTablebaseManager(GameClass)

        print(manager.query_position(GameClass.parse_fen('4k3/8/8/7P/8/4K2B/8/8 w - - 0 1'), outcome_only=True))
        print(manager.query_position(GameClass.parse_fen('4k3/8/8/7P/8/4K2B/8/8 b - - 0 1'), outcome_only=True))

    def test_tablebases(self):
        """
        Test all generated tablebases against other existing tablebases.
        The tablebases are downloaded programmatically from tablebase.sesse.net
        The tablebases are loaded and queried using the python-chess library.
        """
        # TODO: currently this is failing, and it seems like the downloaded tablebases are incorrect
        manager = ChessTablebaseManager(GameClass)
        for descriptor in GameClass.DRAWING_DESCRIPTORS + manager.available_tablebases:
            if len(descriptor) > 2:
                python_chess_descriptor = TestChessTablebase.to_python_chess_descriptor(descriptor)
                TestChessTablebase.ensure_python_chess_loaded(python_chess_descriptor)

        for descriptor in manager.available_tablebases:
            if descriptor == 'KBNk':
                continue
            manager.ensure_loaded(descriptor)
            tablebase = manager.tablebases[descriptor]

            python_chess_tablebase = chess.syzygy.open_tablebase('test_data')

            for board_bytes, move_bytes in tablebase.items():
                fen = GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))
                fen = fen[:-3] + '1 1'
                _, outcome, _ = GameClass.parse_move_bytes(move_bytes)
                board = chess.Board()
                board.set_fen(fen)
                python_chess_outcome = python_chess_tablebase.probe_wdl_table(board)
                python_chess_outcome = np.sign(python_chess_outcome) * (1 if board.turn == chess.WHITE else -1)
                if outcome != python_chess_outcome:
                    raise AssertionError(f'Failed for fen: {fen}')

            manager.clear_tablebases()

    @staticmethod
    def to_python_chess_descriptor(descriptor):
        beginning = ''.join([letter for letter in descriptor if letter.isupper()])
        ending = ''.join([letter for letter in descriptor if letter.islower()])
        return f'{beginning}v{ending.upper()}'

    @staticmethod
    def ensure_python_chess_loaded(python_chess_descriptor):
        for extension in ['rtbw', 'rtbz']:
            if not os.path.isfile(f'test_data/{python_chess_descriptor}.{extension}'):
                # noinspection HttpUrlsUsage
                url = f'http://tablebase.sesse.net/syzygy/3-4-5/{python_chess_descriptor}.{extension}'

                print(f'Downloading file: {python_chess_descriptor}.{extension}')
                with urllib.request.urlopen(url) as f:
                    tablebase = f.read()
                with open(f'test_data/{python_chess_descriptor}.{extension}', 'wb') as file:
                    file.write(tablebase)


if __name__ == '__main__':
    unittest.main()
