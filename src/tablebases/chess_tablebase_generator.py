import pickle
import numpy as np
from games.chess import Chess, encode_fen, parse_fen, PIECE_LETTERS
from tablebases.tablebase_manager import TablebaseManager
from tablebases.symmetry_transform import SymmetryTransform
from utils.utils import OptionalPool
from functools import partial


class ChessTablebaseGenerator:
    """
    Tablebases will always be generated with white as the side who is up in material.
    If material is equal, then it will be treated as if white is up in material.
    Material comparison is determined using Chess.heuristic.

    Symmetry will be applied to ensure that for the attacking king: i < 4, j < 4, i <=j.

    The tablebase will not support any positions where pawns are present or castling is possible.
    """
    class Node:
        def __init__(self, state, tablebase_manager):
            # since many nodes will be stored in memory at once during generation, only the fen will be stored
            self.fen = encode_fen(state)
            self.is_maximizing = Chess.is_player_1_turn(state)
            self.optimizer = max if self.is_maximizing else min

            # create children, but leave references to other nodes as move_fen strings for now
            self.children = []
            self.children_symmetry_transforms = []
            piece_count = np.sum(state[:, :, :12] == 1)
            moves = Chess.get_possible_moves(state)
            for move in moves:
                move_piece_count = np.sum(move[:, :, :12] == 1)
                if move_piece_count == piece_count:
                    symmetry_transform = SymmetryTransform(move)
                    move_fen = encode_fen(symmetry_transform.transform_state(move))
                    self.children.append(move_fen)
                    self.children_symmetry_transforms.append(symmetry_transform)
                else:
                    node = ChessTablebaseGenerator.Node(move, tablebase_manager)
                    node.children = []
                    node.outcome, node.terminal_distance = tablebase_manager.query_position(move, outcome_only=True)
                    self.children.append(node)
                    self.children_symmetry_transforms.append(SymmetryTransform.identity())

            self.best_move = None
            self.best_symmetry_transform = None
            self.outcome = Chess.get_winner(state, moves) if Chess.is_over(state, moves) else None
            self.terminal_distance = 0 if self.outcome is not None else None

        def init_children(self, nodes):
            # replace move_fen strings with references to actual nodes
            self.children = [nodes[child] if type(child) is str else child for child in self.children]

        def update(self):
            """
            :return: True if an update was made.
            """
            terminal_children = [(child, symmetry_transform)
                                 for child, symmetry_transform in zip(self.children, self.children_symmetry_transforms)
                                 if child.outcome is not None]

            if len(terminal_children) == 0:
                return False

            losing_outcome = -1 if self.is_maximizing else 1
            best_move, best_symmetry_transform = terminal_children[0]
            for move, symmetry_transform in terminal_children[1:]:
                if (move.outcome > best_move.outcome) if self.is_maximizing else (move.outcome < best_move.outcome):
                    best_move, best_symmetry_transform = move, symmetry_transform
                elif move.outcome == best_move.outcome:
                    if (move.terminal_distance < best_move.terminal_distance) if move.outcome != losing_outcome \
                            else (move.terminal_distance > best_move.terminal_distance):
                        best_move, best_symmetry_transform = move, symmetry_transform

            updated = False
            if best_move != self.best_move:
                self.best_move = best_move
                self.best_symmetry_transform = best_symmetry_transform
                updated = True
            if self.outcome != best_move.outcome:
                self.outcome = best_move.outcome
                updated = True
            if self.terminal_distance != best_move.terminal_distance + 1:
                self.terminal_distance = best_move.terminal_distance + 1
                updated = True
            return updated

        def get_best_move(self):
            if self.best_move is None:
                return None

            transformed_move = parse_fen(self.best_move.fen)
            best_move_fen = encode_fen(self.best_symmetry_transform.untransform_state(transformed_move))
            return best_move_fen

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
                yield indices.copy()

            for pointer in range(piece_count - 1, 0, -1):  # intentionally skip index 0
                indices[pointer] = indices[pointer][0], indices[pointer][1] + 1
                if indices[pointer][1] == Chess.COLUMNS:
                    indices[pointer] = indices[pointer][0] + 1, 0
                    if indices[pointer][0] == Chess.ROWS:
                        indices[pointer] = 0, 0
                        continue
                break
            else:
                unique_squares_index += 1
                if unique_squares_index == len(SymmetryTransform.UNIQUE_SQUARE_INDICES):
                    break
                indices[0] = SymmetryTransform.UNIQUE_SQUARE_INDICES[unique_squares_index]

    @classmethod
    def create_node(cls, piece_indices, pieces, tablebase_manager):
        nodes = {}
        for is_white_turn in True, False:
            # create the state
            state = np.zeros(Chess.STATE_SHAPE)
            for (i, j), k in zip(piece_indices, pieces):
                state[i, j, k] = 1
            if is_white_turn:
                state[:, :, -1] = 1

            # check if the state is illegal because the player whose turn it isn't is in check
            friendly_slice, enemy_slice, pawn_direction, *_ = Chess.get_stats(state)
            # We need to flip the slices and pawn_direction because the stats are computed for the move,
            # not the state preceding it
            if not Chess.king_safe(state, enemy_slice, friendly_slice, -pawn_direction):
                continue

            node = ChessTablebaseGenerator.Node(state, tablebase_manager)
            nodes[node.fen] = node
        return nodes

    @classmethod
    def generate_tablebase(cls, descriptor, threads=14):
        pieces = sorted([PIECE_LETTERS.index(letter) for letter in descriptor])
        if len(set(pieces)) < len(pieces):
            raise NotImplementedError('Tablebases with duplicate pieces not implemented!')
        if not (0 in pieces and 6 in pieces):
            raise ValueError('White and black kings must be in the descriptor!')

        nodes = {}
        tablebase_manager = TablebaseManager()
        with OptionalPool(threads) as pool:
            for some_nodes in pool.map(partial(cls.create_node, pieces=pieces, tablebase_manager=tablebase_manager),
                                       cls.piece_position_generator(len(pieces))):
                nodes.update(**some_nodes)

        with open(f'chess_tablebases/{descriptor}_nodes.pickle', 'wb') as file:
            pickle.dump(nodes, file)

        for i, node in enumerate(nodes.values()):
            node.init_children(nodes)

        updated = True
        while updated:
            updated = False
            for node in nodes.values():
                if node.update():
                    updated = True

        tablebase = {node.fen: (node.get_best_move(), node.outcome, node.terminal_distance)
                     for node in nodes.values()}
        with open(f'chess_tablebases/{descriptor}.pickle', 'wb') as file:
            pickle.dump(tablebase, file)


def main():
    # KQk,KRk,KBNk
    # TODO: KBBk
    for descriptor in 'KQkn,KQkb,KQkr'.split(','):
        ChessTablebaseGenerator.generate_tablebase(descriptor)
        print(f'Completed {descriptor}')


if __name__ == '__main__':
    main()
