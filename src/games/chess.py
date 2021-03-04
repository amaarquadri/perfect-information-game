from games.game import Game
import numpy as np
from tablebases.tablebase_manager import TablebaseManager
from utils.utils import one_hot, iter_product, STRAIGHT_DIRECTIONS, DIAGONAL_DIRECTIONS, DIRECTIONS_8
from functools import partial


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
    STARTING_STATE = None  # defined after this class's definition
    STATE_SHAPE = (8, 8, 14)  # 8, 8, 14
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
    WHITE_SLICE = slice(0, 6)
    BLACK_SLICE = slice(6, 12)
    PIECE_LETTERS = 'KQRBNPkqrbnp'

    @classmethod
    def parse_algebraic_notation(cls, square, layer_slice=None, as_slice=True):
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

    @classmethod
    def parse_fen(cls, fen):
        pieces, turn, castling, en_passant, *_ = fen.split(' ')
        for i in range(2, 9):
            pieces = pieces.replace(str(i), i * '1')

        def parse_piece(piece_character):
            if piece_character == '1':
                return np.zeros(len(cls.PIECE_LETTERS))
            where = np.argwhere(piece_character == np.array(list(cls.PIECE_LETTERS)))
            if len(where) == 0:
                raise ValueError(f'Invalid piece: {piece_character}')
            return one_hot(where[0, 0], len(cls.PIECE_LETTERS))

        pieces = np.stack([np.stack([parse_piece(piece) for piece in rank], axis=0)
                           for rank in pieces.split('/')], axis=0)

        special_moves = np.zeros((8, 8, 1))
        for letter, square in zip('KQkq', ['G1', 'C1', 'G8', 'C8']):
            if letter in castling:
                special_moves[cls.parse_algebraic_notation(square)] = 1
        if en_passant != '-':
            special_moves[cls.parse_algebraic_notation(en_passant.capitalize())] = 1

        turn = np.full((8, 8, 1), 1 if turn.lower() == 'w' else 0)
        return np.concatenate((pieces, special_moves, turn), axis=-1).astype(np.uint8)

    @classmethod
    def encode_fen(cls, state):
        def encode_piece(piece_arr):
            where = np.argwhere(piece_arr)
            if len(where) == 0:
                return '1'
            if len(where) > 1:
                raise ValueError('Multiple pieces in same square')
            return cls.PIECE_LETTERS[where[0, 0]]

        pieces = '/'.join([''.join([encode_piece(state[rank, file, :12]) for file in range(8)]) for rank in range(8)])
        for i in range(8, 1, -1):
            pieces = pieces.replace(i * '1', str(i))

        turn = 'w' if Chess.is_player_1_turn(state) else 'b'

        castling = ''
        for letter, square in [('K', 'G1'), ('Q', 'C1'), ('k', 'G8'), ('q', 'C8')]:
            if state[cls.parse_algebraic_notation(square, layer_slice=-2)] == 1:
                castling += letter
        if len(castling) == 0:
            castling = '-'

        en_passant = None
        for letter in 'ABCDEFGH':
            for number in '36':
                square = letter + number
                if state[cls.parse_algebraic_notation(square, layer_slice=-2)] == 1:
                    if en_passant is None:
                        en_passant = square.lower()
                    else:
                        raise ValueError(f'Multiple en passant flags detected: {en_passant}, {square}')
        if en_passant is None:
            en_passant = '-'

        return ' '.join([pieces, turn, castling, en_passant, '-', '-'])

    @classmethod
    def encode_board_bytes(cls, state):
        # https://codegolf.stackexchange.com/a/19446
        bitboard = np.sum(state[:, :, :12], axis=-1) == 1
        is_white_turn = cls.is_player_1_turn(state)
        pieces = []
        for i, j in iter_product(Chess.BOARD_SHAPE):
            if not bitboard[i, j]:
                continue

            # KQRBNP, King to move, Rook that can castle or Pawn that moved 2 squares, same 8 for black
            where = np.argwhere(state[i, j, :12])[0, 0]
            piece = where % 6
            is_white = where < 6

            if piece == 0 and is_white == is_white_turn:  # if piece is a king whose turn it is
                pieces.append(6 if is_white_turn else 14)
                continue

            if piece == 2 and i == (7 if is_white else 0):  # if piece is a rook on home rank
                if i == 0 and j == 0 and state[0, 2, -2] == 1:
                    pieces.append(15)
                    continue
                if i == 0 and j == 7 and state[0, 6, -2] == 1:
                    pieces.append(15)
                    continue
                if i == 7 and j == 0 and state[7, 2, -2] == 1:
                    pieces.append(7)
                    continue
                if i == 7 and j == 7 and state[7, 6, -2] == 1:
                    pieces.append(7)
                    continue

            if piece == 5 and is_white != is_white_turn and i == (4 if is_white else 3) \
                    and state[(5 if is_white else 2), j, -2] == 1:
                pieces.append(7 if is_white else 15)
                continue

            pieces.append(piece + (0 if is_white else 8))

        board_bytes = []
        for row in bitboard:
            value = 0
            for i, square in enumerate(row):
                if square:
                    value += (1 << i)
            board_bytes.append(value)
        if np.sum(bitboard) % 2 == 1:
            board_bytes.append(pieces[0])
            pieces = pieces[1:]
        for i in range(0, len(pieces), 2):
            board_bytes.append(16 * pieces[i] + pieces[i + 1])
        return bytes(board_bytes)

    @classmethod
    def parse_board_bytes(cls, board_bytes):
        board_bytes = list(board_bytes)  # convert back to list of integers
        bitboard = np.array([[row & (1 << square) for square in range(8)] for row in board_bytes[:8]]) != 0
        board_bytes = board_bytes[8:]
        pieces = []
        piece_count = np.sum(bitboard)
        if piece_count % 2 == 1:
            pieces.append(board_bytes[0])
            board_bytes = board_bytes[1:]
        for piece_bytes in board_bytes:
            pieces.append(piece_bytes // 16)
            pieces.append(piece_bytes % 16)

        if len(pieces) != piece_count:
            raise ValueError(f'Inconsistent number of pieces! Expected {piece_count} but got {len(pieces)}')

        state = np.zeros(Chess.STATE_SHAPE)
        en_passant_processed = False
        for i, j in iter_product(Chess.BOARD_SHAPE):
            if not bitboard[i, j]:
                continue

            piece_value = pieces.pop(0)
            is_white = piece_value < 8
            piece = piece_value % 8

            if piece == 6:  # if the piece is a king whose turn it is
                # process turn information
                if is_white:
                    state[:, :, -1] = 1
                piece = 0  # convert to regular king
            if piece == 7:
                if i == 0 or i == 7:  # if this is castling information
                    if i == 0 and j == 0 and not is_white:
                        state[0, 2, -2] = 1
                    elif i == 0 and j == 7 and not is_white:
                        state[0, 6, -2] = 1
                    elif i == 7 and j == 0 and is_white:
                        state[7, 2, -2] = 1
                    elif i == 7 and j == 7 and is_white:
                        state[7, 6, -2] = 1
                    else:
                        raise ValueError(f'Castling information on invalid square! '
                                         f'i = {i}, j = {j}, is_white = {is_white}')
                    piece = 2  # convert to regular rook
                elif i == 3 or i == 4:  # if this is en passant information
                    if en_passant_processed:
                        raise ValueError(f'Second en passant square found! i = {i}, j = {j}, is_white = {is_white}')
                    if i == 4 and is_white:
                        state[i + 1, j, -2] = 1
                    elif i == 3 and not is_white:
                        state[i - 1, j, -2] = 1
                    else:
                        raise ValueError(f'En passant rank does not match the correct colour! '
                                         f'i = {i}, j = {j}, is_white = {is_white}')
                    en_passant_processed = True
                    piece = 5  # convert to regular pawn
                else:
                    raise ValueError(f'Special piece on invalid square! i = {i}, j = {j}, is_white = {is_white}')

            state[i, j, piece + (0 if is_white else 6)] = 1

        if en_passant_processed:
            # check if en passant information is consistent with whose turn it is
            is_white_pawn_en_passant = np.any(state[5, :, -2] == 1)
            if is_white_pawn_en_passant == cls.is_player_1_turn(state):
                raise ValueError('The player whose turn it is also has an en passant pawn!')

        return state

    def __init__(self, state=STARTING_STATE):
        # noinspection PyTypeChecker
        super().__init__(self.parse_fen(state) if type(state) is str else state)

    def perform_user_move(self, clicks):
        (start_i, start_j), (end_i, end_j) = clicks
        self.state = self.apply_from_to_move(self.state, start_i, start_j, end_i, end_j)

    @classmethod
    def apply_from_to_move(cls, state, start_i, start_j, end_i, end_j):
        friendly_slice, enemy_slice, *_ = cls.get_stats(state)

        for move in cls.get_possible_moves(state):
            if np.any(state[start_i, start_j, friendly_slice] == 1) and \
                    np.all(state[end_i, end_j, friendly_slice] == 0) and \
                    np.all(move[start_i, start_j, :12] == 0) and \
                    np.any(move[end_i, end_j, friendly_slice] == 1):
                return move
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
        legal_moves = np.full(cls.MOVE_SHAPE, False)
        friendly_slice, *_ = cls.get_stats(state)
        for move in cls.get_possible_moves(state):
            start_i, start_j, end_i, end_j = cls.get_from_to_move(state, move, friendly_slice)
            legal_moves[start_i, start_j, end_i, end_j] = True
        return legal_moves

    @classmethod
    def get_from_to_move(cls, state, move, friendly_slice=None):
        if friendly_slice is None:
            friendly_slice, *_ = cls.get_stats(state)

        from_squares = []  # squares that went from friendly to empty
        to_squares = []  # squares that went from not friendly to friendly
        for i, j, in iter_product(cls.BOARD_SHAPE):
            if np.any(state[i, j, friendly_slice] == 1) and np.all(move[i, j, :12] == 0):
                # insert to the start of the list if its a king
                if state[i, j, friendly_slice][0] == 1:
                    from_squares.insert(0, (i, j))
                else:
                    from_squares.append((i, j))
            elif np.all(state[i, j, friendly_slice] == 0) and np.any(move[i, j, friendly_slice] == 1):
                # insert to the start of the list if its a king
                if move[i, j, friendly_slice][0] == 1:
                    to_squares.insert(0, (i, j))
                else:
                    to_squares.append((i, j))

        if (len(from_squares) == 1 and len(to_squares) == 1) or (len(from_squares) == 2 and len(to_squares) == 2):
            return from_squares[0][0], from_squares[0][1], to_squares[0][0], to_squares[0][1]
        else:
            raise Exception(f'Invalid number of piece moves: from_squares = {from_squares}, to_squares = {to_squares}')

    @classmethod
    def get_stats(cls, state):
        if cls.is_player_1_turn(state):
            friendly_slice = cls.WHITE_SLICE
            enemy_slice = cls.BLACK_SLICE
            pawn_direction = -1
            queening_row = 0
            pawn_starting_row = 6
            castling_row = 7
            en_passant_row = 3
        else:
            friendly_slice = cls.BLACK_SLICE
            enemy_slice = cls.WHITE_SLICE
            pawn_direction = 1
            queening_row = 7
            pawn_starting_row = 1
            castling_row = 0
            en_passant_row = 4
        return friendly_slice, enemy_slice, pawn_direction, queening_row, pawn_starting_row, castling_row, \
            en_passant_row

    @classmethod
    def is_draw_by_insufficient_material(cls, state):
        return TablebaseManager.get_position_descriptor(cls, state) in TablebaseManager.DRAWING_DESCRIPTORS

    @classmethod
    def is_over(cls, state, moves=None):
        return cls.is_draw_by_insufficient_material(state) or \
               (len(cls.get_possible_moves(state)) == 0 if moves is None else len(moves) == 0)

    @classmethod
    def get_winner(cls, state, moves=None):
        if not cls.is_over(state, moves):
            raise Exception('Game is not over!')

        if cls.is_draw_by_insufficient_material(state):
            return 0

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
        return np.sum(np.dot(state, [100, 9, 5, 3.25, 3, 1, -100, -9, -5, -3.25, -3, -1, 0, 0]))


# need to define this after the Chess class is created so that we can use the parse_fen function
Chess.STARTING_STATE = Chess.parse_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
