# Use Gan: one network choosing placement, one shooting
# generator network maps a vector from a random state space to the output choice of piece placement
from .game import Game


class Battleship(Game):
    def perform_user_move(self, clicks):
        pass

    @classmethod
    def get_possible_moves(cls, state):
        pass

    @classmethod
    def get_legal_moves(cls, state):
        pass

    @classmethod
    def is_over(cls, state):
        pass

    @classmethod
    def get_winner(cls, state):
        pass
