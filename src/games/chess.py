from games.game import Game
import numpy as np
from utils.utils import one_hot, iter_product, STRAIGHT_DIRECTIONS, DIAGONAL_DIRECTIONS, DIRECTIONS_8
from functools import partial


def parse_algebraic_notation(square, layer_slice=None, as_slice=True):
    letter, number = square

    where = np.argwhere(np.array(list('87654321')) == number)
    if len(where) == 0:
        raise ValueError(f'Invalid number: {number}')
    i = where[0, 0]

    where = np.argwhere(np.array(list('ABCDEFGH')) == letter)
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


def parse_fen(position):
    pieces, turn, castling, en_passant, *_ = position.split(' ')
    for i in range(2, 9):
        pieces = pieces.replace(str(i), i * '1')

    mapping = 'KQRBNPkqrbnp'

    def parse_piece(piece_character):
        if piece_character == '1':
            return np.zeros(len(mapping))
        where = np.argwhere(piece_character == np.array(list(mapping)))
        if len(where) == 0:
            raise ValueError(f'Invalid piece: {piece_character}')
        return one_hot(where[0, 0], len(mapping))

    pieces = np.stack([np.stack([parse_piece(piece) for piece in rank], axis=0)
                       for rank in pieces.split('/')], axis=0)

    special_moves = np.zeros((8, 8, 1))
    for letter, square in zip('KQkq', ['C1', 'G1', 'C8', 'G8']):
        if letter in castling:
            special_moves[parse_algebraic_notation(square)] = 1
    if en_passant != '-':
        special_moves[parse_algebraic_notation(en_passant.capitalize())] = 1

    turn = np.full((8, 8, 1), 1 if turn.lower() == 'w' else 0)
    return np.concatenate((pieces, special_moves, turn), axis=-1)


def encode_fen(state):
    mapping = 'KQRBNPkqrbnp'

    def encode_piece(piece_arr):
        where = np.argwhere(piece_arr)
        if len(where) == 0:
            return '1'
        if len(where) > 1:
            raise ValueError('Multiple pieces in same square')
        return mapping[where[0, 0]]
    pieces = '/'.join([''.join([encode_piece(state[rank, file, :12]) for file in range(8)]) for rank in range(8)])
    for i in range(8, 1, -1):
        pieces = pieces.replace(i * '1', str(i))

    turn = 'w' if Chess.is_player_1_turn(state) else 'b'

    castling = ''
    for letter, square in [('K', 'G1'), ('Q', 'C1'), ('k', 'G8'), ('q', 'G1')]:
        if state[parse_algebraic_notation(square, layer_slice=-2)] == 1:
            castling += letter
    if len(castling) == 0:
        castling = '-'

    en_passant = None
    for letter in 'ABCDEFGH':
        for number in '36':
            square = letter + number
            if state[parse_algebraic_notation(square, layer_slice=-2)] == 1:
                if en_passant is None:
                    en_passant = square.lower()
                else:
                    raise ValueError(f'Multiple en passant flags detected: {en_passant}, {square}')
    if en_passant is None:
        en_passant = '-'

    return ' '.join([pieces, turn, castling, en_passant, '-', '-'])


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
    STARTING_STATE = parse_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    STATE_SHAPE = STARTING_STATE.shape  # 8, 8, 14
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 8, 8, 14
    BOARD_SHAPE = (ROWS, COLUMNS)  # 8, 8
    MOVE_SHAPE = (ROWS, COLUMNS, ROWS, COLUMNS)  # 8, 8, 8, 8: start_i, start_j, end_i, end_j
    REPRESENTATION_LETTERS = ['K', 'Q', 'R', 'B', 'N', 'P',
                              'k', 'q', 'r', 'b', 'n', 'p']
    CLICKS_PER_MOVE = 2
    REPRESENTATION_FILES = [None,
                            'white_king', 'white_queen', 'white_rook', 'white_bishop', 'white_knight', 'white_pawn',
                            'black_king', 'black_queen', 'black_rook', 'black_bishop', 'black_knight', 'black_pawn']
    KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

    def __init__(self, state=STARTING_STATE):
        super().__init__(parse_fen(state) if type(state) is str else state)

    def perform_user_move(self, clicks):
        (start_i, start_j), (end_i, end_j) = clicks
        friendly_slice, enemy_slice, *_ = self.get_stats(self.state)

        for move in self.get_possible_moves(self.state):
            if np.any(self.state[start_i, start_j, friendly_slice] == 1) and \
                    np.all(self.state[end_i, end_j, friendly_slice] == 0) and \
                    np.all(move[start_i, start_j, :12] == 0) and \
                    np.any(move[end_i, end_j, friendly_slice] == 1):
                self.state = move
                break
        else:
            raise ValueError('Invalid Move!')

    @classmethod
    def needs_checkerboard(cls):
        return True

    @classmethod
    def create_move(cls, state, i, j, target_i, target_j):
        move = cls.null_move(state)
        move[target_i, target_j, :12] = move[i, j, :12]
        move[i, j, :12] = 0
        # handle loss of castling privileges due to rook move or capture
        for castling_row in [0, 7]:
            for castling_column, rook_pos in [(2, 0), (6, 7)]:
                if state[castling_row, castling_column, -2] == 1:
                    # only need to check if the rook moves or is captured
                    # if the king moves, then castling privileges will be removed as part of get_possible_moves
                    if (i == castling_row and j == rook_pos) or \
                            (target_i == castling_row and target_j == rook_pos):
                        move[castling_row, castling_column, -2] = 0
        return move

    @classmethod
    def get_infinite_distance_moves(cls, state, i, j, directions, friendly_slice):
        moves = []
        for di, dj in directions:
            for dist in range(1, 8):
                target_i, target_j = i + dist * di, j + dist * dj
                if not cls.is_valid(target_i, target_j):
                    break
                if np.all(state[target_i, target_j, friendly_slice] == 0):
                    moves.append(cls.create_move(state, i, j, target_i, target_j))
                if np.any(state[target_i, target_j, :12] == 1):
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
                    king_moves = cls.get_finite_distance_moves(state, i, j, DIRECTIONS_8, friendly_slice)

                    # castling moves
                    king_column = 4
                    for castling_column, pass_through_column, rook_column, empty_column in [(2, 3, 0, 1),
                                                                                            (6, 5, 7, None)]:
                        # don't need to check that castling_column is safe because that will be done later anyways
                        if state[castling_row, castling_column, -2] and \
                                np.all(state[castling_row, pass_through_column, :12] == 0) and \
                                np.all(state[castling_row, castling_column, :12] == 0) and \
                                cls.square_safe(state, castling_row, king_column, enemy_slice, -pawn_direction) and \
                                cls.square_safe(state, castling_row, pass_through_column, enemy_slice, -pawn_direction) and \
                                state[castling_row, rook_column, friendly_slice][2] == 1 and \
                                (empty_column is None or np.all(state[castling_row, empty_column, :12] == 0)):
                            move = cls.create_move(state, i, j, castling_row, castling_column)
                            move[castling_row, pass_through_column, :12] = move[castling_row, rook_column, :12]
                            move[castling_row, rook_column, :12] = 0
                            king_moves.append(move)

                    # remove the castling flag from all moves
                    for move in king_moves:
                        move[castling_row, :, -2] = 0

                    moves.extend(king_moves)

                if square_piece[1] == 1:  # queen moves
                    moves.extend(cls.get_infinite_distance_moves(state, i, j, DIRECTIONS_8, friendly_slice))

                if square_piece[2] == 1:  # rook moves
                    rook_moves = cls.get_infinite_distance_moves(state, i, j, STRAIGHT_DIRECTIONS, friendly_slice)

                    # if castling was possible before the rook moved, remove the castling flag from all moves
                    for castling_column, rook_column in [(2, 0), (6, 7)]:
                        if state[castling_row, castling_column, -2] == 1 and i == castling_row and j == rook_column:
                            for move in rook_moves:
                                move[castling_row, castling_column, -2] = 0

                    moves.extend(rook_moves)

                if square_piece[3] == 1:  # bishop moves
                    moves.extend(cls.get_infinite_distance_moves(state, i, j, DIAGONAL_DIRECTIONS, friendly_slice))

                if square_piece[4] == 1:  # knight moves
                    moves.extend(cls.get_finite_distance_moves(state, i, j, cls.KNIGHT_MOVES, friendly_slice))

                if square_piece[5] == 1:  # pawn moves
                    is_promoting = i + pawn_direction == queening_row
                    if np.all(state[i + pawn_direction, j, :12] == 0):
                        move = cls.create_move(state, i, j, i + pawn_direction, j)
                        if is_promoting:
                            moves.extend(cls.promote_on_move(move, i + pawn_direction, j, friendly_slice))
                        else:
                            moves.append(move)

                    if i == pawn_starting_row and \
                            np.all(state[i + pawn_direction, j, :12] == 0) and \
                            np.all(state[i + 2 * pawn_direction, j, :12] == 0):
                        move = cls.create_move(state, i, j, i + 2 * pawn_direction, j)

                        # set en passant flag if there is an adjacent enemy pawn
                        for dj in [1, -1]:
                            if cls.is_valid(i + 2 * pawn_direction, j + dj) and \
                                    state[i + 2 * pawn_direction, j + dj, enemy_slice][5] == 1:
                                # play out the en passant capture and verify that it is a valid move
                                test_board = cls.null_move(move)
                                test_move = cls.create_move(test_board, i + 2 * pawn_direction, j + dj,
                                                            i + pawn_direction, j)
                                test_move[i + 2 * pawn_direction, j, :12] = 0
                                if cls.king_safe(test_move, enemy_slice, friendly_slice, -pawn_direction):
                                    # verified, set the en passant flag now
                                    move[i + pawn_direction, j, -2] = 1
                                    break

                        moves.append(move)

                    for dj in [1, -1]:
                        target_i, target_j = i + pawn_direction, j + dj
                        if cls.is_valid(target_i, target_j):
                            if np.any(state[target_i, target_j, enemy_slice] == 1):
                                move = cls.create_move(state, i, j, target_i, target_j)
                                if is_promoting:
                                    moves.extend(cls.promote_on_move(move, target_i, target_j, friendly_slice))
                                else:
                                    moves.append(move)

                            elif i == en_passant_row and state[target_i, target_j, -2] == 1 \
                                    and state[i, target_j, enemy_slice][5] == 1 \
                                    and np.all(state[target_i, target_j, :12] == 0):
                                move = cls.create_move(state, i, j, target_i, target_j)
                                move[i, target_j, :12] = 0
                                moves.append(move)

        king_safe_func = partial(cls.king_safe,
                                 friendly_slice=friendly_slice,  enemy_slice=enemy_slice, pawn_direction=pawn_direction)
        return list(filter(king_safe_func, moves))

    @classmethod
    def square_safe(cls, state, i, j, attacking_slice, attacking_pawn_direction):
        """

        :param state:
        :param i:
        :param j:
        :param attacking_slice: The pieces that are potentially attacking the square of interest.
        :param attacking_pawn_direction: The direction that the attacking pieces' pawns move.
        :return:
        """
        for directions, relevant_pieces in [(STRAIGHT_DIRECTIONS, [1, 2]),
                                            (DIAGONAL_DIRECTIONS, [1, 3])]:
            for di, dj in directions:
                for dist in range(1, 8):
                    target_i, target_j = i + dist * di, j + dist * dj
                    if not cls.is_valid(target_i, target_j):
                        break
                    if dist == 1 and state[target_i, target_j, attacking_slice][0]:
                        return False  # enemy king is adjacent to this square
                    if np.any(state[target_i, target_j, attacking_slice][relevant_pieces] == 1):
                        return False  # enemy queen, rook, or bishop
                    if np.any(state[target_i, target_j, :12] == 1):
                        break
        for di, dj in cls.KNIGHT_MOVES:
            target_i, target_j = i + di, j + dj
            if cls.is_valid(target_i, target_j) and state[target_i, target_j, attacking_slice][4] == 1:
                return False  # enemy knight
        for dj in [-1, 1]:
            target_i, target_j = i - attacking_pawn_direction, j + dj
            if cls.is_valid(target_i, target_j) and state[target_i, target_j, attacking_slice][5] == 1:
                return False  # enemy pawn
        return True

    @classmethod
    def get_king_pos(cls, state, player_slice):
        king_pos = None
        for i, j in iter_product(cls.BOARD_SHAPE):
            if state[i, j, player_slice][0] == 1:
                if king_pos is None:
                    king_pos = i, j
                else:
                    raise ValueError('Multiple kings found!')
        if king_pos is None:
            raise ValueError('No king found!')
        return king_pos

    @classmethod
    def king_safe(cls, move, friendly_slice, enemy_slice, pawn_direction):
        """
        Returns False if the provided move results in a state that is not valid because the king would be in check.

        :returns: True if and only if the player whose turn it isn't has a king that is safe.
        """
        king_i, king_j = cls.get_king_pos(move, friendly_slice)
        return cls.square_safe(move, king_i, king_j, enemy_slice, -pawn_direction)

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
            queening_row = 7
            pawn_starting_row = 1
            castling_row = 0
            en_passant_row = 4
        return friendly_slice, enemy_slice, pawn_direction, queening_row, pawn_starting_row, castling_row, \
            en_passant_row

    @classmethod
    def is_over(cls, state):
        return len(cls.get_possible_moves(state)) == 0

    @classmethod
    def get_winner(cls, state):
        if not cls.is_over(state):
            raise Exception('Game is not over!')

        friendly_slice, enemy_slice, pawn_direction, *_ = cls.get_stats(state)
        king_i, king_j = cls.get_king_pos(state, friendly_slice)
        if cls.square_safe(state, king_i, king_j, enemy_slice, -pawn_direction):
            return 0  # stalemate

        if cls.is_player_1_turn(state):
            return -1
        else:
            return 1

    @classmethod
    def is_empty(cls, state, i, j):
        return np.all(state[i, j, :-2] == 0)

    @classmethod
    def null_move(cls, state):
        move = super(Chess, cls).null_move(state)
        # remove en passant possibilities
        move[2, :, -2] = 0
        move[-3, :, -2] = 0
        return move

    @staticmethod
    def is_board_full(state):
        return False

    @classmethod
    def heuristic(cls, state):
        return np.sum(np.dot(state, [100, 9, 5, 3.25, 3, 1, 100, -9, -5, -3.25, -3, -1, 0, 0]))
