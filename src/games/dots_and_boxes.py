from .game import Game


class DotsAndBoxes(Game):
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
