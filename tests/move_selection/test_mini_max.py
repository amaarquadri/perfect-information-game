import unittest
from move_selection.mini_max import MiniMax
from games.tic_tac_toe import TicTacToe as GameClass


class TestMiniMax(unittest.TestCase):
    def test_solver(self):
        mini_max = MiniMax.solver(GameClass)
        print(mini_max.generate_random_game())


if __name__ == '__main__':
    unittest.main()
