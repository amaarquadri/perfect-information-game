import unittest
from perfect_information_game.move_selection import RandomMoveChooser
from perfect_information_game.games import Checkers as GameClass


class TestRandomMoveChooser(unittest.TestCase):
    def test_generate_random_game(self):
        move_chooser = RandomMoveChooser(GameClass)
        print(move_chooser.generate_random_game(max_moves=5))


if __name__ == '__main__':
    unittest.main()
