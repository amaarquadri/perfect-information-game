import unittest
from src.move_selection.random_chooser import RandomMoveChooser
from src.games.checkers import Checkers as GameClass


class TestRandomMoveChooser(unittest.TestCase):
    def test_generate_random_game(self):
        move_chooser = RandomMoveChooser(GameClass)
        print(move_chooser.generate_random_game(max_moves=5))


if __name__ == '__main__':
    unittest.main()
