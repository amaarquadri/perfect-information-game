import unittest
from perfect_information_game.move_selection import MiniMax
from perfect_information_game.games import TicTacToe as GameClass


class TestMiniMax(unittest.TestCase):
    def test_solver(self):
        mini_max = MiniMax.solver(GameClass)
        training_data, outcome = mini_max.generate_random_game()
        if outcome != 0:
            raise AssertionError('Tic tac toe solver did not draw the game!')
        print(training_data, outcome)


if __name__ == '__main__':
    unittest.main()
