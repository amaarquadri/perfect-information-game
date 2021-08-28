import pickle
import numpy as np
from perfect_information_game.tablebases import ChessTablebaseManager, SymmetryTransform, get_verified_chess_subclass
from perfect_information_game.utils import get_training_path
from functools import partial


class ChessTablebaseGenerator:
    """
    Tablebases will always be generated with white as the side who is up in material.
    If material is equal, then it will be treated as if white is up in material.
    Material comparison is determined using GameClass.heuristic.

    If there are no pawns, symmetry will be applied to ensure that for the attacking king: i < 4, j < 4, i <= j.
    If there are pawns, symmetry will be applied to ensure that for the attacking king: j < 4.

    The tablebase will not support any positions where en passant or castling is possible.
    """
    class Node:
        def __init__(self, GameClass, state, descriptor, tablebase_manager):
            self.GameClass = get_verified_chess_subclass(GameClass)

            # since many nodes will be stored in memory at once during generation, only the board_bytes will be stored
            self.board_bytes = self.GameClass.encode_board_bytes(state)
            self.is_maximizing = self.GameClass.is_player_1_turn(state)
            heuristic = self.GameClass.heuristic(state)
            self.has_material_advantage = (heuristic > 0) if self.is_maximizing else (heuristic < 0)

            self.children = []
            self.children_symmetry_transforms = []
            self.best_move = None
            self.best_symmetry_transform = None

            moves = self.GameClass.get_possible_moves(state)
            if self.GameClass.is_over(state, moves):
                self.outcome = self.GameClass.get_winner(state, moves)
                self.terminal_distance = 0
                return  # skip populating children
            else:
                # assume that everything is a draw (by fortress) unless proven otherwise
                # this will get overwritten with a win if any child node is proven to be a win
                # this will get overwritten with a loss if all child nodes are proven to be a loss
                self.outcome = 0
                self.terminal_distance = np.inf

            # populate children, but leave references to other nodes as board_bytes for now
            for move in moves:
                # need to compare descriptors (piece count is not robust to pawn promotions)
                move_descriptor = self.GameClass.get_position_descriptor(move)
                if move_descriptor == descriptor:
                    symmetry_transform = SymmetryTransform(self.GameClass, move)
                    move_board_bytes = self.GameClass.encode_board_bytes(symmetry_transform.transform_state(move))
                    self.children.append(move_board_bytes)
                    self.children_symmetry_transforms.append(symmetry_transform)
                else:
                    node = ChessTablebaseGenerator.Node(self.GameClass, move, move_descriptor, tablebase_manager)
                    node.children = []
                    node.outcome, node.terminal_distance = tablebase_manager.query_position(move, outcome_only=True)
                    if np.isnan(node.outcome) or np.isnan(node.terminal_distance):
                        raise ValueError(
                            f'Could not find position in existing tablebase! descriptor = {move_descriptor}')
                    self.children.append(node)
                    self.children_symmetry_transforms.append(SymmetryTransform.identity(self.GameClass))

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
                    # pick the faster win if both moves are winning, and pick the slower loss if both moves are losing
                    elif (move.terminal_distance < best_move.terminal_distance) if move.outcome != losing_outcome \
                            else (move.terminal_distance > best_move.terminal_distance):
                        best_move, best_symmetry_transform = move, symmetry_transform

            updated = False
            if best_move != self.best_move:
                self.best_move = best_move
                self.best_symmetry_transform = best_symmetry_transform
                # don't update for draws because of potential infinite loops (although that shouldn't happen in theory)
                # TODO: figure this out and remove it
                if self.outcome != 0:
                    updated = True
            if self.outcome != best_move.outcome:
                self.outcome = best_move.outcome
                updated = True
            if self.terminal_distance != best_move.terminal_distance + 1:
                # note that this arithmetic will work, even with np.inf
                self.terminal_distance = best_move.terminal_distance + 1
                # don't update for draws because of potential infinite loops (although that shouldn't happen in theory)
                # TODO: figure this out and remove it
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

        def get_best_move_data(self):
            """
            The node's destroy_connections function must be called first.
            """
            if self.best_move is None:
                # this node was terminal
                return None

            if type(self.best_move) is not bytes:
                print('Warning: destroy_connections not called. Calling now...')
                self.destroy_connections()

            transformed_move = self.GameClass.parse_board_bytes(self.best_move)
            move = self.best_symmetry_transform.untransform_state(transformed_move)
            return self.GameClass.get_from_to_move(self.GameClass.parse_board_bytes(self.board_bytes), move)

        @staticmethod
        def get_move_bytes(node, GameClass):
            return GameClass.encode_move_bytes(node.get_best_move_data(), node.outcome, node.terminal_distance)

    def __init__(self, GameClass):
        self.GameClass = get_verified_chess_subclass(GameClass)
        self.tablebase_manager = ChessTablebaseManager(self.GameClass)

    def piece_config_generator(self, descriptor):
        """
        This function is a generator that iterates over all possible piece configurations.
        Each piece configuration yielded is a list of (i, j, k) tuples,
        where (i, j) represents the location of the piece and k represents the piece itself.

        The first element of the yielded lists will be the attacking king,
        which will only be placed on squares in the unique squares defined by
        SymmetryTransform.UNIQUE_SQUARE_INDICES or SymmetryTransform.PAWNLESS_UNIQUE_SQUARE_INDICES.

        The descriptor must have white as the side who is up in material (or equal).
        """
        pieces = np.array([self.GameClass.PIECE_LETTERS.index(letter) for letter in descriptor])
        if np.sum(pieces == self.GameClass.WHITE_KING) != 1 or np.sum(pieces == self.GameClass.BLACK_KING) != 1:
            raise ValueError('Descriptor must have exactly 1 white king and 1 black king!')

        piece_count = len(pieces)
        pawnless = 'p' not in descriptor and 'P' not in descriptor
        unique_squares = SymmetryTransform.PAWNLESS_UNIQUE_SQUARE_INDICES if pawnless \
            else SymmetryTransform.UNIQUE_SQUARE_INDICES
        unique_squares_index = 0

        piece_config = [unique_squares[unique_squares_index] + (self.GameClass.KING, )] + \
                       [(0, 0, piece) for piece in pieces if piece != self.GameClass.KING]

        yielded_configurations = set()

        while True:
            if len(set(map(lambda piece_i_config: piece_i_config[:2], piece_config))) == piece_count and \
                    tuple(piece_config) not in yielded_configurations:
                # only yield if there are no pieces on the same square
                # and an identical configuration has not been yielded before
                yield piece_config.copy()
                yielded_configurations.add(tuple(piece_config))

            for pointer in range(piece_count - 1, 0, -1):  # intentionally skip index 0 because it is handled separately
                piece_config[pointer] = piece_config[pointer][0], piece_config[pointer][1] + 1, piece_config[pointer][2]
                if piece_config[pointer][1] == self.GameClass.COLUMNS:
                    piece_config[pointer] = piece_config[pointer][0] + 1, 0, piece_config[pointer][2]
                    if piece_config[pointer][0] == self.GameClass.ROWS:
                        piece_config[pointer] = 0, 0, piece_config[pointer][2]
                        continue
                break
            else:
                unique_squares_index += 1
                if unique_squares_index == len(unique_squares):
                    break
                piece_config[0] = unique_squares[unique_squares_index] + (0,)

    def create_nodes(self, piece_config, descriptor):
        nodes = {}
        for is_white_turn in (True, False):
            # create the state
            state = np.zeros(self.GameClass.STATE_SHAPE)
            for i, j, k in piece_config:
                state[i, j, k] = 1
            if is_white_turn:
                state[:, :, -1] = 1

            # check if the state is illegal because the player whose turn it isn't is in check
            friendly_slice, enemy_slice, pawn_direction, *_ = self.GameClass.get_stats(state)
            # We need to flip the slices and pawn_direction because the stats are computed for the move,
            # not the state preceding it
            if not self.GameClass.king_safe(state, enemy_slice, friendly_slice, -pawn_direction):
                continue

            if np.any(state[[0, self.GameClass.ROWS - 1], :, self.GameClass.WHITE_PAWN] == 1) or \
                    np.any(state[[0, self.GameClass.ROWS - 1], :, self.GameClass.BLACK_PAWN] == 1):
                # ignore states with pawns on the first or last ranks
                continue

            node = ChessTablebaseGenerator.Node(self.GameClass, state, descriptor, self.tablebase_manager)
            nodes[node.board_bytes] = node
        return nodes

    def generate_tablebase(self, descriptor, pool):
        nodes_path = f'{get_training_path(self.GameClass)}/tablebases/nodes/{descriptor}_nodes.pickle'
        try:
            with open(nodes_path, 'rb') as file:
                nodes = pickle.load(file)
            print(f'Using existing nodes file: {nodes_path}')
        except FileNotFoundError:
            nodes = {}
            for some_nodes in pool.map(partial(self.create_nodes, descriptor=descriptor),
                                       self.piece_config_generator(descriptor)):
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
        node_move_bytes = pool.map(partial(ChessTablebaseGenerator.Node.get_move_bytes, GameClass=self.GameClass),
                                   nodes)
        tablebase = {node.board_bytes: move_bytes for node, move_bytes in zip(nodes, node_move_bytes)
                     # this check ensures that terminal nodes are not included
                     if move_bytes is not None}

        with open(f'{get_training_path(self.GameClass)}/tablebases/{descriptor}.pickle', 'wb') as file:
            pickle.dump(tablebase, file)

        self.tablebase_manager.update_tablebase_list()
