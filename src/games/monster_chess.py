from games.chess import Chess


class MonsterChess(Chess):
    """
    https://en.wikipedia.org/wiki/Monster_chess
    White has only a king and the center 4 pawns on their usual squares, whereas black has all their usual pieces.
    However, white makes 2 moves for every move that black makes.

    The state
    """
    DRAWING_DESCRIPTORS = []

    @classmethod
    def is_draw_by_insufficient_material(cls, state):
        return False
