import unittest
import numpy as np
from games.checkers import Checkers


class TestCheckers(unittest.TestCase):
    def test_get_possible_moves(self):
        # starting state
        moves_from_start = Checkers.get_possible_moves(Checkers.STARTING_STATE)
        self.assertEqual(len(moves_from_start), 7)
        for move in moves_from_start:
            self.assertEqual(Checkers.is_player_1_turn(move), False)

        # random test position
        moves = Checkers.get_possible_moves(np.array([
            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1]],

            [[0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]],

            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]],

            [[0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]],

            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]],

            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]],

            [[0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1]],

            [[1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1]]]))
        # 1 move is a double jump
        self.assertEqual(sum([Checkers.is_player_1_turn(move) for move in moves]), 1)
        self.assertEqual(sum([not Checkers.is_player_1_turn(move) for move in moves]), 6)
        self.assertEqual(sum([move[0, 1, 1] == 1 for move in moves]), 1)  # check that 1 move results in a king


if __name__ == '__main__':
    unittest.main()
