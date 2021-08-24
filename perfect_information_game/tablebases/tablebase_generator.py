import pickle
import numpy as np
from perfect_information_game.tablebases import TablebaseManager
from perfect_information_game.tablebases import SymmetryTransform
from perfect_information_game.utils import get_training_path
from perfect_information_game.tablebases import get_verified_chess_subclass
from functools import partial


class TablebaseGenerator:
    """
    Tablebases will always be generated with white as the side who is up in material.
    If material is equal, then it will be treated as if white is up in material.
    Material comparison is determined using GameClass.heuristic.

    Symmetry will be applied to ensure that for the attacking king: i < 4, j < 4, i <= j.

    The tablebase will not support any positions where pawns are present or castling is possible.
    """
    class Node:
        def __init__(self, GameClass, state, descriptor, tablebase_manager):
            self.GameClass = get_verified_chess_subclass(GameClass)

            # since many nodes will be stored in memory at once during generation, only the fen will be stored
            self.board_bytes = self.GameClass.encode_board_bytes(state)
            self.is_maximizing = self.GameClass.is_player_1_turn(state)
            heuristic = self.GameClass.heuristic(state)
            self.has_material_advantage = (heuristic > 0) if self.is_maximizing else (heuristic < 0)

            # create children, but leave references to other nodes as move_fen strings for now
            self.children = []
            self.children_symmetry_transforms = []
            moves = self.GameClass.get_possible_moves(state)
            for move in moves:
                # need to compare descriptors (piece count is not robust to pawn promotions)
                move_descriptor = self.GameClass.get_position_descriptor(move)
                if move_descriptor == descriptor:
                    symmetry_transform = SymmetryTransform(self.GameClass, move)
                    move_board_bytes = self.GameClass.encode_board_bytes(symmetry_transform.transform_state(move))
                    self.children.append(move_board_bytes)
                    self.children_symmetry_transforms.append(symmetry_transform)
                else:
                    node = TablebaseGenerator.Node(self.GameClass, move, move_descriptor, tablebase_manager)
                    node.children = []
                    node.outcome, node.terminal_distance = tablebase_manager.query_position(move, outcome_only=True)
                    if np.isnan(node.outcome) or np.isnan(node.terminal_distance):
                        raise ValueError(
                            f'Could not find position in existing tablebase! descriptor = {move_descriptor}')
                    self.children.append(node)
                    self.children_symmetry_transforms.append(SymmetryTransform.identity(self.GameClass))

            # TODO: remove reliance on the fact that:
            #  if self.GameClass.is_over(state) then len(self.GameClass.get_possible_moves(state)) == 0
            self.best_move = None
            self.best_symmetry_transform = None
            if self.GameClass.is_over(state, moves):
                self.outcome = self.GameClass.get_winner(state, moves)
                self.terminal_distance = 0
            else:
                # assume that everything is a draw (by fortress) unless proven otherwise
                # this will get overwritten with a win if any child node is proven to be a win
                # this will get overwritten with a loss if all child nodes are proven to be a loss
                self.outcome = 0
                self.terminal_distance = np.inf

        def init_children(self, nodes):
            # replace move_board_bytes with references to actual nodes
            self.children = [nodes[child] if type(child) is bytes else child for child in self.children]

        def update(self):
            """
            :return: True if an update was made.
            """
            if len(self.children) == 0:
                # this was a terminal node from the start, so there is nothing to update
                return False

            losing_outcome = -1 if self.is_maximizing else 1
            best_move, best_symmetry_transform = self.children[0], self.children_symmetry_transforms[0]
            for move, symmetry_transform in zip(self.children[1:], self.children_symmetry_transforms[1:]):
                if (move.outcome > best_move.outcome) if self.is_maximizing else (move.outcome < best_move.outcome):
                    best_move, best_symmetry_transform = move, symmetry_transform
                elif move.outcome == best_move.outcome:
                    if move.outcome == 0:
                        # TODO: fix bugs in draw logic
                        #  AI is losing KRkr (regular chess) from some drawn positions
                        # if we are up in material, try to delay the draw
                        # if we are down in material (or equal), try to hasten the draw
                        if (move.terminal_distance > best_move.terminal_distance) if self.has_material_advantage \
                                else (move.terminal_distance < best_move.terminal_distance):
                            best_move, best_symmetry_transform = move, symmetry_transform
                    elif (move.terminal_distance < best_move.terminal_distance) if move.outcome != losing_outcome \
                            else (move.terminal_distance > best_move.terminal_distance):
                        best_move, best_symmetry_transform = move, symmetry_transform

            updated = False
            if best_move != self.best_move:
                self.best_move = best_move
                self.best_symmetry_transform = best_symmetry_transform
                # don't update for draws because of potential infinite loops (although that shouldn't happen in theory)
                if self.outcome != 0:
                    updated = True
            if self.outcome != best_move.outcome:
                self.outcome = best_move.outcome
                updated = True
            if self.terminal_distance != best_move.terminal_distance + 1:
                # note that this arithmetic will work, even with np.inf
                self.terminal_distance = best_move.terminal_distance + 1
                # don't update for draws because of potential infinite loops (although that shouldn't happen in theory)
                if self.outcome != 0:
                    updated = True
            return updated

        def check_recursion(self):
            """
            Performs a sanity check on the results and prints a warning if inconsistencies are found.
            
            :return:
            """
            seen_positions = set()
            move = self
            while move is not None:
                if move.board_bytes in seen_positions:
                    if self.terminal_distance < np.inf:
                        print('Warning: terminal distance was finite despite infinite loop!')
                        self.terminal_distance = np.inf
                    return True  # reached infinite loop
                else:
                    seen_positions.add(move.board_bytes)
                move = self.best_move
            return False

        def destroy_connections(self):
            """
            Deletes links to children, and replaces best_move link with a copy of the best_move's board_bytes
            This allows this node to be pickled (without infinite nested references),
            which allows the final operations (calculation of move_bytes) to be done in parallel.
            """
            self.children = None
            self.children_symmetry_transforms = None
            self.best_move = self.best_move.board_bytes if self.best_move is not None else None

        def get_best_move(self):
            """
            The node's destroy_connections function must be called first.
            """
            if self.best_move is None:
                return 0, 0, 0, 0

            if type(self.best_move) is not bytes:
                print('Warning: destroy_connections not called. Calling now...')
                self.destroy_connections()

            transformed_move = self.GameClass.parse_board_bytes(self.best_move)
            move = self.best_symmetry_transform.untransform_state(transformed_move)
            return self.GameClass.get_from_to_move(self.GameClass.parse_board_bytes(self.board_bytes), move)

        @staticmethod
        def get_move_bytes(node):
            start_i, start_j, end_i, end_j = node.get_best_move()
            return TablebaseManager.encode_move_bytes(node.outcome, start_i, start_j, end_i, end_j,
                                                      node.terminal_distance)

    def __init__(self, GameClass):
        self.GameClass = get_verified_chess_subclass(GameClass)
        self.tablebase_manager = TablebaseManager(self.GameClass)

    def piece_position_generator(self, piece_count, pawnless=True):
        """
        This function is a generator that iterates over all possible piece configurations.
        Each piece configuration yielded is a list of (i, j) tuples.
        The first index corresponds to the attacking king, which will only be placed on squares in the 10 unique squares
        defined by SymmetryTransform.UNIQUE_SQUARE_INDICES.
        This function assumes that pieces are unique. For example, having 2 white rooks are not supported.

        :param piece_count: The number of pieces.
        :param pawnless: If True, then the fact that there are no pawns will be used to impose extra symmetries and
                         decrease the total number of nodes to be considered.
        """
        # TODO: handle repeated pieces
        unique_squares_index = 0
        unique_squares = SymmetryTransform.PAWNLESS_UNIQUE_SQUARE_INDICES if pawnless \
            else SymmetryTransform.UNIQUE_SQUARE_INDICES
        indices = [unique_squares[unique_squares_index]] + [(0, 0)] * (piece_count - 1)
        while True:
            # yield if all indices are unique and attacking king is in one of the
            if len(set(indices)) == piece_count:
                yield indices.copy()

            for pointer in range(piece_count - 1, 0, -1):  # intentionally skip index 0
                indices[pointer] = indices[pointer][0], indices[pointer][1] + 1
                if indices[pointer][1] == self.GameClass.COLUMNS:
                    indices[pointer] = indices[pointer][0] + 1, 0
                    if indices[pointer][0] == self.GameClass.ROWS:
                        indices[pointer] = 0, 0
                        continue
                break
            else:
                unique_squares_index += 1
                if unique_squares_index == len(unique_squares):
                    break
                indices[0] = unique_squares[unique_squares_index]

    def create_node(self, piece_indices, descriptor, pieces):
        nodes = {}
        for is_white_turn in True, False:
            # create the state
            state = np.zeros(self.GameClass.STATE_SHAPE)
            for (i, j), k in zip(piece_indices, pieces):
                state[i, j, k] = 1
            if is_white_turn:
                state[:, :, -1] = 1

            # check if the state is illegal because the player whose turn it isn't is in check
            friendly_slice, enemy_slice, pawn_direction, *_ = self.GameClass.get_stats(state)
            # We need to flip the slices and pawn_direction because the stats are computed for the move,
            # not the state preceding it
            if not self.GameClass.king_safe(state, enemy_slice, friendly_slice, -pawn_direction):
                continue

            if np.any(state[0, :, 5] == 1) or np.any(state[7, :, 5] == 1) \
                    or np.any(state[0, :, 11] == 1) or np.any(state[7, :, 11] == 1):
                # ignore states with pawns on the first or last ranks
                continue

            node = TablebaseGenerator.Node(self.GameClass, state, descriptor, self.tablebase_manager)
            nodes[node.board_bytes] = node
        return nodes

    def generate_tablebase(self, descriptor, pool):
        pieces = sorted([self.GameClass.PIECE_LETTERS.index(letter) for letter in descriptor])
        if len(set(pieces)) < len(pieces):
            raise NotImplementedError('Tablebases with duplicate pieces not implemented!')
        if not (0 in pieces and 6 in pieces):
            raise ValueError('White and black kings must be in the descriptor!')

        nodes_path = f'{get_training_path(self.GameClass)}/tablebases/{descriptor}_nodes.pickle'
        try:
            with open(nodes_path, 'rb') as file:
                nodes = pickle.load(file)
            print(f'Using existing nodes file: {nodes_path}')
        except FileNotFoundError:
            nodes = {}
            pawnless = 'P' not in descriptor and 'p' not in descriptor
            for some_nodes in pool.map(partial(self.create_node, descriptor=descriptor, pieces=pieces),
                                       self.piece_position_generator(len(pieces), pawnless)):
                nodes.update(some_nodes)

            with open(nodes_path, 'wb') as file:
                pickle.dump(nodes, file)

        for i, node in enumerate(nodes.values()):
            node.init_children(nodes)

        updated = True
        iterations = 0
        while updated:
            updated = False
            for node in nodes.values():
                if node.update():
                    updated = True
            iterations += 1
            if iterations % 10 == 0:
                print(f'{iterations} iterations completed')

        nodes = list(nodes.values())
        for node in nodes:
            node.destroy_connections()
        node_move_bytes = pool.map(TablebaseGenerator.Node.get_move_bytes, nodes)
        tablebase = {node.board_bytes: move_bytes for node, move_bytes in zip(nodes, node_move_bytes)}

        with open(f'{get_training_path(self.GameClass)}/tablebases/{descriptor}.pickle', 'wb') as file:
            pickle.dump(tablebase, file)

        self.tablebase_manager.update_tablebase_list()

# create a copy of ChessTablebaseGenerator with the legacy name to allow unpickling
# TablebaseGenerator = ChessTablebaseGenerator
