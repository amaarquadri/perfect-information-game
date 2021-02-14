from games.game import Game
import numpy as np
from utils.utils import iter_product, STRAIGHT_DIRECTIONS, DIAGONAL_DIRECTIONS, DIRECTIONS_8


def parse_algebraic_notation(square, layer_slice=None, as_slice=True):
    letter, number = square

    where = np.argwhere(np.array(list('87654321')) == number)
    if len(where) == 0:
        raise ValueError(f'Invalid number: {number}')
    i = where[0, 0]

    where = np.argwhere(np.array(list('ABCFEDGH')) == letter)
    if len(where) == 0:
        raise ValueError(f'Invalid letter: {letter}')
    j = where[0, 0]

    if as_slice:
        if layer_slice is None:
            layer_slice = slice(None)
        if type(layer_slice) is int:
            layer_slice = slice(layer_slice, layer_slice + 1)
        return slice(i, i + 1), slice(j, j + 1), layer_slice
    else:
        return i, j


def create_white_starting_position():
    white = np.zeros((8, 8, 6))
    white[parse_algebraic_notation('E1', 0)] = 1
    white[parse_algebraic_notation('D1', 1)] = 1
    white[parse_algebraic_notation('A1', 2)] = 1
    white[parse_algebraic_notation('H1', 2)] = 1
    white[parse_algebraic_notation('C1', 3)] = 1
    white[parse_algebraic_notation('F1', 3)] = 1
    white[parse_algebraic_notation('B1', 4)] = 1
    white[parse_algebraic_notation('G1', 4)] = 1
    white[-2, :, 5] = 1  # all of row 2
    return white


def create_starting_special_moves():
    special_moves = np.zeros((8, 8, 1))
    special_moves[parse_algebraic_notation('C1', 4)] = 1
    special_moves[parse_algebraic_notation('G1', 4)] = 1
    special_moves[parse_algebraic_notation('C8', 4)] = 1
    special_moves[parse_algebraic_notation('G8', 4)] = 1
    return special_moves


class Chess(Game):
    """
    The game state is represented by an 8x8x14 matrix. The 14 layers correspond to:
    white king, white queens, white rooks, white bishops, white knights, white pawns,
    black king, black queens, black rooks, black bishops, black knights, black pawns,
    special moves, whose turn.
    Threefold repetition and the 50 move rule are not included.

    The special moves layer will have a 1 on any square that contains a square where the king will end up after a legal
    castling move (C1, G1, C8, and G8), or the square where a pawn will end up after a legal
    en passant capture (rows 3 and 6).
    """
    WHITE = create_white_starting_position()
    BLACK = np.flip(WHITE, axis=0)
    SPECIAL_MOVES = create_starting_special_moves()
    STARTING_STATE = np.concatenate([WHITE, BLACK, SPECIAL_MOVES, np.ones((8, 8, 1))], axis=-1).astype(np.uint8)
    STATE_SHAPE = STARTING_STATE.shape  # 8, 8, 14
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 8, 8, 14
    BOARD_SHAPE = (ROWS, COLUMNS)  # 8, 8
    MOVE_SHAPE = (ROWS, COLUMNS, ROWS, COLUMNS)  # 8, 8, 8, 8: start_i, start_j, end_i, end_j
    REPRESENTATION_LETTERS = ['K', 'Q', 'R', 'B', 'N', 'P',
                              'k', 'q', 'r', 'b', 'n', 'p']
    CLICKS_PER_MOVE = 2
    REPRESENTATION_FILES = ['light_square', 'dark_square',
                            'white_king_light_square', 'white_king_dark_square',
                            'white_queen_light_square', 'white_queen_dark_square',
                            'white_rook_light_square', 'white_rook_dark_square',
                            'white_bishop_light_square', 'white_bishop_dark_square',
                            'white_knight_light_square', 'white_knight_dark_square',
                            'white_pawn_light_square', 'white_pawn_dark_square',
                            'black_king_light_square', 'black_king_dark_square',
                            'black_queen_light_square', 'black_queen_dark_square',
                            'black_rook_light_square', 'black_rook_dark_square',
                            'black_bishop_light_square', 'black_bishop_dark_square',
                            'black_knight_light_square', 'black_knight_dark_square',
                            'black_pawn_light_square', 'black_pawn_dark_square']
    KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -1), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        (start_i, start_j), (end_i, end_j) = clicks
        # TODO: implement

    @classmethod
    def create_move(cls, state, i, j, target_i, target_j):
        move = cls.null_move(state)
        move[target_i, target_j, :12] = move[i, j, :12]
        move[i, j, :12] = 0
        return move

    @classmethod
    def get_infinite_distance_moves(cls, state, i, j, directions, friendly_slice, enemy_slice):
        moves = []
        for di, dj in directions:
            for dist in range(1, 8):
                target_i, target_j = i + dist * di, j + dist * dj
                if not cls.is_valid(target_i, target_j):
                    break
                if np.all(state[target_i, target_j, friendly_slice] == 0):
                    moves.append(cls.create_move(state, i, j, target_i, target_j))
                if np.any(state[target_i, target_j, enemy_slice] == 1):
                    break
        return moves

    @classmethod
    def get_finite_distance_moves(cls, state, i, j, displacements, friendly_slice):
        moves = []
        for di, dj in displacements:
            target_i, target_j = i + di, j + dj
            if cls.is_valid(target_i, target_j) and np.all(state[target_i, target_j, friendly_slice] == 0):
                moves.append(cls.create_move(state, i, j, target_i, target_j))
        return moves

    @classmethod
    def promote_on_move(cls, move, target_i, target_j, friendly_slice):
        promoting_moves = []
        for promotion in range(1, 5):
            promoting_move = np.copy(move)
            promoting_move[target_i, target_j, :12] = 0
            promoting_move[target_i, target_j, friendly_slice][promotion] = 1
            promoting_moves.append(promoting_move)
        return promoting_moves

    @classmethod
    def get_possible_moves(cls, state):
        friendly_slice, enemy_slice, pawn_direction, queening_row, pawn_starting_row, castling_row, en_passant_row = \
            cls.get_stats(state)
        moves = []

        for i, j in iter_product(cls.BOARD_SHAPE):
            square_piece = state[i, j, friendly_slice]
            if np.any(square_piece == 1):
                if square_piece[0] == 1:  # king moves
                    king_moves = cls.get_finite_distance_moves(i, j, DIRECTIONS_8, friendly_slice)

                    # castling moves
                    for castling_column, pass_through_column, rook_column in [(2, 3, 0), (6, 5, 7)]:
                        if state[castling_row, castling_column, -2] and \
                                np.all(state[castling_row, pass_through_column, :12] == 0) and \
                                np.all(state[castling_row, castling_column, :12] == 0) and \
                                cls.square_safe(state, castling_row, 4) and \
                                cls.square_safe(state, castling_row, pass_through_column):
                            move = cls.create_move(state, i, j, castling_row, castling_column)
                            move[castling_row, pass_through_column, :12] = move[castling_row, rook_column, :12]
                            move[castling_row, rook_column, :12] = 0
                            king_moves.append(move)

                    # remove the castling flag from all moves
                    for move in king_moves:
                        move[castling_row, :, -2] = 0

                    moves.extend(king_moves)

                if square_piece[1] == 1:  # queen moves
                    moves.extend(cls.get_infinite_distance_moves(i, j, DIRECTIONS_8, friendly_slice, enemy_slice))

                if square_piece[2] == 1:  # rook moves
                    rook_moves = cls.get_infinite_distance_moves(i, j, STRAIGHT_DIRECTIONS, friendly_slice, enemy_slice)

                    # if castling was possible before the rook moved, remove the castling flag from all moves
                    for castling_column, rook_column in [(2, 0), (6, 7)]:
                        if state[castling_row, castling_column, -2] == 1 and i == castling_row and j == rook_column:
                            for move in rook_moves:
                                move[castling_row, castling_column, -2] = 0

                    moves.extend(rook_moves)

                if square_piece[3] == 1:  # bishop moves
                    moves.extend(cls.get_infinite_distance_moves(i, j, DIAGONAL_DIRECTIONS, friendly_slice, enemy_slice))

                if square_piece[4] == 1:  # knight moves
                    moves.extend(cls.get_finite_distance_moves(i, j, cls.KNIGHT_MOVES, friendly_slice))

                if square_piece[5] == 1:  # pawn moves
                    is_promoting = i + pawn_direction == queening_row
                    if np.all(state[i + pawn_direction, j, :12] == 0):
                        move = cls.create_move(state, i, j, i + pawn_direction, j)
                        if is_promoting:
                            moves.extend(cls.promote_on_move(move, i + pawn_direction, j, friendly_slice))
                        else:
                            moves.append(move)

                    if i == pawn_starting_row and np.all(state[i + 2 * pawn_direction, j, :12] == 0):
                        move = cls.create_move(state, i, j, i + 2 * pawn_direction, j)

                        # set en passant flag if there is an adjacent enemy pawn
                        for dj in [1, -1]:
                            if cls.is_valid(i + 2 * pawn_direction, j + dj) and \
                                    state[i + 2 * pawn_direction, j + dj, enemy_slice][5] == 1:
                                move[i + pawn_direction, j, -2] = 1
                                break

                        moves.append(move)

                    for dj in [1, -1]:
                        if cls.is_valid(i + pawn_direction, j + dj):
                            if np.any(state[i + pawn_direction, j + dj, enemy_slice] == 1):
                                move = cls.create_move(state, i, j, i + pawn_direction, j + dj)
                                if is_promoting:
                                    moves.extend(cls.promote_on_move(move, i + pawn_direction, j + dj, friendly_slice))
                                else:
                                    moves.append(move)

                            elif i == en_passant_row and state[i + pawn_direction, j + dj, -2] == 1:
                                move = cls.create_move(state, i, j, i + pawn_direction, j + dj)
                                move[i, j + dj, :12] = 0
                                moves.append(move)

        return list(filter(cls.king_safe, moves))

    @classmethod
    def square_safe(cls, state, i, j):
        return True

    @classmethod
    def king_safe(cls, move):
        return True

    @classmethod
    def get_legal_moves(cls, state):
        # TODO: implement
        return []

    @classmethod
    def get_stats(cls, state):
        if cls.is_player_1_turn(state):
            friendly_slice = slice(0, 6)
            enemy_slice = slice(6, 12)
            pawn_direction = -1
            queening_row = 0
            pawn_starting_row = 6
            castling_row = 7
            en_passant_row = 3
        else:
            friendly_slice = slice(6, 12)
            enemy_slice = slice(0, 6)
            pawn_direction = 1
            queening_row = 8
            pawn_starting_row = 1
            castling_row = 0
            en_passant_row = 4
        return friendly_slice, enemy_slice, pawn_direction, queening_row, pawn_starting_row, castling_row, \
               en_passant_row

    @classmethod
    def is_over(cls, state):
        # TODO: implement
        return False

    @classmethod
    def get_winner(cls, state):
        # TODO: implement
        return 0

    @classmethod
    def get_img_index_representation(cls, state):
        # same calculation as in Game, except with only half the number of REPRESENTATION_FILES
        representation = np.zeros_like(cls.BOARD_SHAPE)
        for i in range(len(cls.REPRESENTATION_FILES) // 2 - 1):
            representation[state[:, :, i] == 1] = i + 1

        # multiply by 2, add 1 if dark square
        representation *= 2
        representation += np.array([[(i + j % 2) for j in range(8)] for i in range(8)])
        return representation

    @classmethod
    def is_empty(cls, state, i, j):
        return np.all(state[i, j, :-2] == 0)

    @classmethod
    def null_move(cls, state):
        move = super(Chess, cls).null_move(state)
        # remove en passant possibilities
        move[1, :, -2] = 0
        move[-2, :, -2] = 0
        return move

    @staticmethod
    def is_board_full(state):
        return False

    @classmethod
    def heuristic(cls, state):
        return np.sum(np.dot(state, [0, 9, 5, 3, 3, 1, 0, -9, -5, -3, -3, -1, 0, 0]))
