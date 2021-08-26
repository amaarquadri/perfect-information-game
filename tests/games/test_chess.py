import unittest
import json
import numpy as np
from time import time
from perfect_information_game.games import Chess
from perfect_information_game.utils import OptionalPool, iter_product


class TestChess(unittest.TestCase):
    def test_move_counts(self, state, depth=3, threads=8):
        move_counts = []
        moves = [state]
        with OptionalPool(threads) as pool:
            for _ in range(depth):
                moves = sum(pool.map(Chess.get_possible_moves, moves), start=[])
                move_counts.append(len(moves))
        return move_counts

    def test_get_possible_moves(self):
        """
        Test cases taken from https://gist.github.com/peterellisjones/8c46c28141c162d1d8a0f0badbc9cff9

        :return:
        """
        with open('chess_test_cases.json') as f:
            test_cases = sorted(json.load(f), key=lambda case: case['nodes'])

        for test_case in test_cases:
            if test_case['nodes'] < 1000_000:
                node_count = self.search_for_errors_recursive(test_case['fen'], test_case['depth'])
                if node_count != test_case['nodes']:
                    print(test_case['fen'])
                    raise AssertionError
                else:
                    print('pass ' + test_case['fen'])

    def test_chess_programming_cases(self):
        """
        Test cases taken from https://www.chessprogramming.org/Perft_Results
        """
        self.assertEqual(self.test_move_counts(Chess.STARTING_STATE, depth=3), [20, 400, 8902])

        self.assertEqual(self.test_move_counts(Chess.parse_fen(
            'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -'), depth=3),
            [48, 2039, 97862])

        self.assertEqual(self.test_move_counts(Chess.parse_fen('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - '), depth=4),
                         [14, 191, 2812, 43238])

        self.assertEqual(self.test_move_counts(Chess.parse_fen(
            'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1'), depth=3),
            [6, 264, 9467])

        self.assertEqual(self.test_move_counts(Chess.parse_fen(
            'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8'), depth=3),
            [44, 1486, 62379])

        self.assertEqual(self.test_move_counts(
            Chess.parse_fen('r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 '), depth=3),
            [46, 2079, 89890])

    def test(self):
        print(self.search_for_errors_recursive('rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8', depth=3))

    def search_for_errors_recursive(self, fen, depth=3):
        """
        Recursively searches the tree of possible moves, starting from the given fen, until the specified depth.
        If a position is found where the calculated number of moves possible differs from that given by the python
        chess library, then an AssertionError is raised.

        :param fen:
        :param depth:
        :return:
        """
        import chess  # import Python chess library for comparison to find bugs

        def get_fen(board, move):
            board.push(move)
            move_fen = board.fen()
            board.pop()
            return move_fen

        if depth == 0:
            return 1

        chess_board = chess.Board(fen)
        chess_moves = sorted([get_fen(chess_board, move) for move in chess_board.generate_legal_moves()])
        moves = sorted([Chess.encode_fen(move) for move in Chess.get_possible_moves(Chess.parse_fen(fen))])

        if len(moves) != len(chess_moves):
            raise AssertionError

        total = 0
        for move in chess_moves:
            total += self.search_for_errors_recursive(move, depth - 1)
        return total

    def test_board_bytes(self):
        self.test_encodings(Chess.STARTING_STATE)

        with open('chess_test_cases.json') as f:
            test_cases = json.load(f)

        for test_case in test_cases:
            fen = test_case['fen']
            state = Chess.parse_fen(fen)
            self.test_encodings(state)
            for move in Chess.get_possible_moves(state):
                self.test_encodings(move)

    def test_encodings(self, state):
        if not np.all(state == Chess.parse_board_bytes(Chess.encode_board_bytes(state))):
            raise AssertionError(f'Failed to consistently process board_bytes: {Chess.encode_fen(state)}')
        if not np.all(state == Chess.parse_fen(Chess.encode_fen(state))):
            raise AssertionError(f'Failed to consistently process fen: {Chess.encode_fen(state)}')

    def test_benchmark_square_safe(self):
        # run once to compile numba function
        Chess.square_safe(Chess.STARTING_STATE, 0, 0, Chess.WHITE_SLICE, -1)

        with open('chess_test_cases.json') as f:
            test_cases = sorted(json.load(f), key=lambda case: case['nodes'])

        test_cases = [test_case for test_case in test_cases if test_case['nodes'] < 1_000]

        def get_positions(position, depth):
            if depth == 0:
                return [position]
            result = []
            for move in Chess.get_possible_moves(position):
                result.extend(get_positions(move, depth - 1))
            return result

        for test_case in test_cases:
            test_case['positions'] = get_positions(Chess.parse_fen(test_case['fen']), test_case['depth'])

        total_cases = 64 * sum([test_case['nodes'] for test_case in test_cases])
        print(f'Starting benchmark with {total_cases} invocations...')
        start_time = time()
        for test_case in test_cases:
            for state in test_case['positions']:
                for i, j in iter_product(Chess.BOARD_SHAPE):
                    Chess.square_safe(state, i, j, Chess.WHITE_SLICE, -1)
        print(time() - start_time)


if __name__ == '__main__':
    unittest.main()
