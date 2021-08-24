from functools import partial
import numpy as np
from perfect_information_game.games import Chess, InvalidMoveException


class MonsterChess(Chess):
    """
    https://en.wikipedia.org/wiki/Monster_chess
    White has only a king and the center 4 pawns on their usual squares, whereas black has all their usual pieces.
    However, white makes 2 moves for every move that black makes.

    The state has an extra feature layer (the second last one) which is all 1's if and only if it is white's first move.
    """
    STARTING_STATE = None  # defined after class definition
    STATE_SHAPE = (8, 8, 15)  # 8, 8, 15
    FEATURE_COUNT = STATE_SHAPE[-1]  # 15
    DRAWING_DESCRIPTORS = []

    @classmethod
    def create_monster_state(cls, chess_state, is_double_move):
        monster_state = np.ones(cls.BOARD_SHAPE) if is_double_move else np.zeros(cls.BOARD_SHAPE)
        return np.concatenate((chess_state[..., :-1],
                               monster_state[..., np.newaxis],
                               chess_state[..., -1][..., np.newaxis]), axis=-1)

    @classmethod
    def decompose_monster_state(cls, monster_state):
        chess_state = np.concatenate((monster_state[..., :-2], monster_state[..., -1][..., np.newaxis]), axis=-1)
        is_double_move = np.all(monster_state[..., -2])
        return chess_state, is_double_move

    @classmethod
    def parse_fen(cls, fen):
        """
        Note that fens are ambiguous for Monster Chess if it is white's move.
        This function will assume that it is always white's first move if the fen indicates that it is white's move.
        """
        chess_state = Chess.parse_fen(fen)
        return cls.create_monster_state(chess_state, cls.is_player_1_turn(chess_state))

    @classmethod
    def encode_fen(cls, state):
        chess_state, is_double_move = cls.decompose_monster_state(state)
        if cls.is_player_1_turn(chess_state) and not is_double_move:
            raise NotImplementedError('Fens are not defined for Monster Chess when it is white\'s second move!')
        return Chess.encode_fen(chess_state)

    @classmethod
    def encode_board_bytes(cls, state):
        """
        An extra byte is added to the end of the regular Chess board bytes.
        The extra byte is True if and only if it is a double move.
        """
        chess_state, is_double_move = cls.decompose_monster_state(state)
        chess_bytes = Chess.encode_board_bytes(chess_state)
        monster_byte = bytes([is_double_move])
        return chess_bytes + monster_byte

    @classmethod
    def parse_board_bytes(cls, board_bytes):
        """
        An extra byte is assumed to be added to the end of the regular Chess board bytes.
        The extra byte is True if and only if it is a double move.
        """
        chess_state = Chess.parse_board_bytes(board_bytes[:-1])
        return cls.create_monster_state(chess_state, board_bytes[-1])

    def __init__(self, state=STARTING_STATE):
        super().__init__(self.parse_fen(state) if type(state) is str else state)

    @classmethod
    def apply_from_to_move(cls, state, start_i, start_j, end_i, end_j, promotion=None,
                           allow_pseudo_legal=True):
        if not allow_pseudo_legal:
            raise ValueError('Pseudo legal moves must be allowed for Monster Chess!')

        chess_state, is_double_move = cls.decompose_monster_state(state)
        move = Chess.apply_from_to_move(chess_state, start_i, start_j, end_i, end_j, promotion,
                                        allow_pseudo_legal=is_double_move)
        if is_double_move:
            # flip back to white's turn
            move = cls.null_move(move)

            if len(super(MonsterChess, cls).get_possible_moves(move)) == 0:
                raise InvalidMoveException(
                    'Invalid Move: White king will not be able to get out of check on the second move!')

        is_now_double_move = not cls.is_player_1_turn(state)
        if is_now_double_move:
            friendly_slice, enemy_slice, pawn_direction, *_ = cls.get_stats(state)
            if not cls.king_safe(move, friendly_slice, enemy_slice, pawn_direction):
                raise InvalidMoveException('Invalid Move: black king can be captured on white\'s second move')
        return cls.create_monster_state(move, is_now_double_move)

    @classmethod
    def get_move_notation(cls, state, move):
        chess_state, _ = cls.decompose_monster_state(state)
        move_chess_state, _ = cls.decompose_monster_state(move)
        return Chess.get_move_notation(chess_state, move_chess_state)

    @classmethod
    def get_possible_moves(cls, state):
        chess_state, is_double_move = cls.decompose_monster_state(state)

        if is_double_move:
            moves = cls.get_pseudo_legal_moves(chess_state)

            # flip back to white's turn
            moves = [cls.null_move(move) for move in moves]

            # filter out moves that don't permit the king to leave check next move
            moves = [move for move in moves if len(Chess.get_possible_moves(move)) > 0]
        else:
            moves = Chess.get_possible_moves(chess_state)

            friendly_slice, enemy_slice, pawn_direction, *_ = cls.get_stats(chess_state)
            king_safe_func = partial(cls.king_safe, friendly_slice=friendly_slice,
                                     enemy_slice=enemy_slice, pawn_direction=pawn_direction)
            moves = list(filter(king_safe_func, moves))

        is_now_double_move = not cls.is_player_1_turn(state)
        return [cls.create_monster_state(move, is_now_double_move) for move in moves]

    @classmethod
    def is_draw_by_insufficient_material(cls, state):
        return False

    @classmethod
    def king_safe(cls, move, friendly_slice, enemy_slice, pawn_direction):
        if not cls.is_player_1_turn(move):
            return Chess.king_safe(move, friendly_slice, enemy_slice, pawn_direction)

        return np.all([Chess.king_safe(cls.null_move(next_move),
                                       friendly_slice, enemy_slice, pawn_direction)
                       for next_move in cls.get_pseudo_legal_moves(move)])

    @classmethod
    def is_check(cls, state):
        if super(MonsterChess, cls).is_check(state):
            return True
        if not cls.is_player_1_turn(state):
            return np.any([Chess.is_check(move)
                           for move in cls.get_pseudo_legal_moves(cls.null_move(state))])
        return False


MonsterChess.STARTING_STATE = MonsterChess.parse_fen('rnbqkbnr/pppppppp/8/8/8/8/2PPPP2/4K3 w kq - 0 1')
