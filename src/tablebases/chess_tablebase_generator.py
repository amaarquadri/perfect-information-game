import numpy as np
from games.chess import Chess, parse_fen, encode_fen, PIECE_LETTERS
from utils.utils import OptionalPool
from tablebases.symmetry_transform import SymmetryTransform


class ChessTablebaseGenerator:
    """
    Tablebases will always be generated with white as the side who is up in material.
    If material is equal, then it will be treated as if white is up in material.
    Material comparison is determined using Chess.heuristic.

    Symmetry will be applied to ensure that for the attacking king: i < 4, j < 4, i <=j.

    The tablebase will not support any positions where pawns are present or castling is possible.
    """
    class Node:
        def __init__(self, state):
            self.fen = encode_fen(state)
            self.is_maximizing = Chess.is_player_1_turn(state)

            self.children_fen = [encode_fen(move) for move in Chess.get_possible_moves(state)]
            self.children_nodes = None

            self.outcome = Chess.get_winner(state) if Chess.is_over(state) else None
            self.terminal_distance = 0 if self.outcome is not None else None
            self.best_move = None

        def init_children(self, nodes):
            self.children_nodes = [nodes[fen] for fen in self.children_fen]

        def extend(self):
            if self.outcome is not None:
                return

            terminal_children = [child for child in self.children_nodes if child.outcome is not None]
            if len(terminal_children) == 0:
                return

    @classmethod
    def piece_position_generator(cls, piece_count):
        """
        This function is a generator that iterates over all possible piece configurations.
        Each piece configuration yielded is a list of (i, j) tuples.
        The first index corresponds to the attacking king, which will only be placed on squares in the 10 unique squares
        defined by SymmetryTransform.UNIQUE_SQUARE_INDICES.
        This function assumes that pieces are unique. For example, having 2 white rooks are not supported.

        :param piece_count: The number of pieces.
        """
        unique_squares_index = 0
        indices = [SymmetryTransform.UNIQUE_SQUARE_INDICES[unique_squares_index]] + [(0, 0)] * (piece_count - 1)
        while True:
            # yield if all indices are unique and attacking king is in one of the
            if len(set(indices)) == piece_count and indices[0] in SymmetryTransform.UNIQUE_SQUARE_INDICES:
                yield indices

            for pointer in range(piece_count - 1, 0, -1):  # intentionally skip index 0
                indices[pointer] = indices[pointer][0], indices[pointer][1] + 1
                if indices[pointer][1] == Chess.COLUMNS:
                    indices[pointer] = indices[pointer][0] + 1, 0
                    if indices[pointer][0] == Chess.ROWS:
                        continue
                break
            else:
                unique_squares_index += 1
                if unique_squares_index == len(SymmetryTransform.UNIQUE_SQUARE_INDICES):
                    break
                indices[0] = SymmetryTransform.UNIQUE_SQUARE_INDICES[unique_squares_index]

    @classmethod
    def generate_KQ_k_tablebase(cls, pieces, file_name=None):
        pieces = sorted(pieces)
        if len(set(pieces)) < len(pieces):
            raise NotImplementedError('Tablebases with duplicate pieces not implemented!')
        if not (0 in pieces and 6 in pieces):
            raise ValueError('White and black kings must be among the pieces!')

        if file_name is None:
            file_name = ''.join([PIECE_LETTERS[i] for i in pieces]) + '.pickle'

        all_nodes = {}
        for is_white_turn in [True, False]:
            for piece_indices in cls.piece_position_generator(len(pieces)):
                # create the state
                state = np.zeros(Chess.STATE_SHAPE)
                for (i, j), k in zip(piece_indices, pieces):
                    state[i, j, k] = 1
                if is_white_turn:
                    state[:, :, -1] = 1

                # check if the state is illegal because the player whose turn it isn't is in check
                friendly_slice, enemy_slice, pawn_direction, *_ = Chess.get_stats(state)
                if not Chess.king_safe(state, friendly_slice, enemy_slice, pawn_direction):
                    continue

                fen = encode_fen(state)
                all_nodes[fen] = ChessTablebaseGenerator.Node(state)




def main():
    ChessTablebaseGenerator.generate_KQ_k_tablebase([0, 1, 6])


if __name__ == '__main__':
    main()
