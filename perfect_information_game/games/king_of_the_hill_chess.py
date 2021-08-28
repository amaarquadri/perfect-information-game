from perfect_information_game.games import Chess


class KingOfTheHillChess(Chess):
    DRAWING_DESCRIPTORS = []

    @classmethod
    def is_draw_by_insufficient_material(cls, state):
        return False

    @classmethod
    def get_possible_moves(cls, state):
        if cls.get_king_of_the_hill_winner(state) is not None:
            return []
        return super(KingOfTheHillChess, cls).get_possible_moves(state)

    @classmethod
    def get_king_of_the_hill_winner(cls, state):
        white_king_i, white_king_j = cls.get_king_pos(state, cls.WHITE_SLICE)
        if white_king_i in [3, 4] and white_king_j in [3, 4]:
            return 1

        black_king_i, black_king_j = cls.get_king_pos(state, cls.BLACK_SLICE)
        if black_king_i in [3, 4] and black_king_j in [3, 4]:
            return -1
        return None

    @classmethod
    def is_over(cls, state):
        return cls.get_king_of_the_hill_winner(state) is not None or super().is_over(state)

    @classmethod
    def get_winner(cls, state):
        king_of_the_hill_winner = cls.get_king_of_the_hill_winner(state)
        return king_of_the_hill_winner if king_of_the_hill_winner is not None else \
            super(KingOfTheHillChess, cls).get_winner(state)
