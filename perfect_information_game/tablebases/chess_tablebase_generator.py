import pickle
import numpy as np
from perfect_information_game.tablebases import ChessTablebaseManager, SymmetryTransform, get_verified_chess_subclass, \
    TablebaseException
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
        def __init__(self, GameClass, state, descriptor, tablebase_manager, require_terminal=False):
            """
            :param GameClass:
            :param state:
            :param descriptor:
            :param tablebase_manager:
            :param require_terminal: If True, then a TablebaseException will be raised if this node is not terminal.
                                     Note that this ensures that the children will never be populated.
            """
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

            self.outcome, self.terminal_distance = tablebase_manager.query_position(state, outcome_only=True)
            if not np.isnan(self.outcome):
                # if position was found in an existing tablebase
                self.terminal = True
                return  # skip populating children
            else:
                if require_terminal:
                    raise TablebaseException(f'Required position with descriptor {descriptor} '
                                             f'was not found in an existing tablebase!')
                self.terminal = False
                # assume that everything is a draw (by fortress) unless proven otherwise
                # this will get overwritten with a win if any child node is proven to be a win
                # this will get overwritten with a loss if all child nodes are proven to be a loss
                self.outcome = 0
                self.terminal_distance = np.inf

            # populate children, but leave references to other nodes as board_bytes for now
            for move in self.GameClass.get_possible_moves(state):
                # need to compare descriptors (piece count is not robust to pawn promotions)
                move_descriptor = self.GameClass.get_position_descriptor(move, pawn_ranks=True)
                if move_descriptor == descriptor:
                    symmetry_transform = SymmetryTransform(self.GameClass, move)
                    move_board_bytes = self.GameClass.encode_board_bytes(symmetry_transform.transform_state(move))
                    self.children.append(move_board_bytes)
                    self.children_symmetry_transforms.append(symmetry_transform)
                else:
                    # create this node with require_terminal=True so that if it is not terminal a TablebaseException
                    # will be raised right away, before its children are populated
                    node = ChessTablebaseGenerator.Node(self.GameClass, move, move_descriptor, tablebase_manager,
                                                        require_terminal=True)
                    self.children.append(node)
                    self.children_symmetry_transforms.append(SymmetryTransform.identity(self.GameClass))

        def init_children(self, nodes):
            # replace move_board_bytes with references to actual nodes
            self.children = [nodes[child] if type(child) is bytes else child for child in self.children]

        def update(self):
            """
            :return: True if an update was made.
            """
            if self.terminal:
                # this was a terminal node from the start
                # (either due to the game ending, or simplification to another descriptor)
                # so there is nothing to update
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
            if self.terminal:
                raise TablebaseException('Cannot get best_move_data from a terminal node!')

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
        if descriptor[0] != self.GameClass.PIECE_LETTERS[self.GameClass.WHITE_KING]:
            raise ValueError('Descriptor must start with white king!')

        def rank_to_row(rank, is_white):
            return 8 - rank if is_white else rank - 1
        pieces = np.array([self.GameClass.PIECE_LETTERS.index(letter)
                           for letter in descriptor if letter.isalpha()])
        restrictions = [rank_to_row(int(descriptor[i + 1]), letter.isupper())
                        if i + 1 < len(descriptor) and descriptor[i + 1].isnumeric() else None
                        for i, letter in enumerate(descriptor) if letter.isalpha()]
        if restrictions[0] is not None:
            raise ValueError('White king position cannot have rank restrictions!')

        if np.sum(pieces == self.GameClass.WHITE_KING) != 1 or np.sum(pieces == self.GameClass.BLACK_KING) != 1:
            raise ValueError('Descriptor must have exactly 1 white king and 1 black king!')

        piece_count = len(pieces)
        pawnless = 'p' not in descriptor and 'P' not in descriptor
        unique_squares = SymmetryTransform.PAWNLESS_UNIQUE_SQUARE_INDICES if pawnless \
            else SymmetryTransform.UNIQUE_SQUARE_INDICES
        unique_squares_index = 0

        piece_config = [unique_squares[unique_squares_index] + (self.GameClass.WHITE_KING, )] + \
                       [(0 if restriction is None else restriction, 0, piece)
                        for piece, restriction in zip(pieces, restrictions)
                        if piece != self.GameClass.WHITE_KING]

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
                    if restrictions[pointer] is not None:
                        piece_config[pointer] = piece_config[pointer][0], 0, piece_config[pointer][2]
                        continue

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
            state = np.zeros(self.GameClass.STATE_SHAPE, dtype=np.uint8)
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

            # TODO: check if en passant might be possible for the piece_config, and compute all results if so
            node = ChessTablebaseGenerator.Node(self.GameClass, state, descriptor, self.tablebase_manager)
            nodes[node.board_bytes] = node
        return nodes

    def generate_tablebase(self, descriptor, pool):
        self.tablebase_manager.update_tablebase_list()
        if descriptor in self.tablebase_manager.available_tablebases:
            raise ValueError('Tablebase for the given descriptor already exists!')

        if 'p' in descriptor and 'P' in descriptor:
            raise NotImplementedError('No support for positions with pawns of both colours yet due to en passant.')

        nodes_path = f'{get_training_path(self.GameClass)}/tablebases/nodes/{descriptor}_nodes.pickle'
        try:
            with open(nodes_path, 'rb') as file:
                nodes = pickle.load(file)
            print(f'Using existing nodes file: {nodes_path}')
        except FileNotFoundError:
            nodes = {}
            """
            Choosing a reasonable chunksize is a complex task.
            Too small and the overhead from inter-process communication for each batch will be too large.
            Too large and the potential lost capacity at the end when some workers have nothing to do will be too large.
            As a rough approximation, we can minimize the following: 
            task_to_overhead_ratio * (threads - 1) * chunksize + tasks / chunksize
            Making the approximations: task_to_overhead_ratio = 500, threads = 15, tasks = 8 * 64 * 64 * 64
            And minimizing gives chunksize ~ 20
            """
            for some_nodes in pool.imap_unordered(partial(self.create_nodes, descriptor=descriptor),
                                                  self.piece_config_generator(descriptor),
                                                  chunksize=20):
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

        nodes = [node for node in nodes.values() if not node.terminal]  # ensure that terminal nodes are not included
        for node in nodes:
            node.destroy_connections()
        node_move_bytes = pool.map(partial(ChessTablebaseGenerator.Node.get_move_bytes, GameClass=self.GameClass),
                                   nodes)
        tablebase = {node.board_bytes: move_bytes for node, move_bytes in zip(nodes, node_move_bytes)}

        with open(f'{get_training_path(self.GameClass)}/tablebases/{descriptor}.pickle', 'wb') as file:
            pickle.dump(tablebase, file)

        self.tablebase_manager.update_tablebase_list()

    @staticmethod
    def generate_descriptors(piece_count):
        if piece_count < 2:
            raise ValueError('Must be at least two pieces!')
        if piece_count == 2:
            yield 'Kk'
            return
        pieces = 'QRBN'
        # values = {'Q': 9, 'R': 5, 'B': 3.25, 'N': 3}
        if piece_count == 3:
            for piece in pieces:
                yield f'K{piece}k'
            for rank in range(7, 1, -1):
                yield f'KP{rank}k'
        if piece_count == 4:
            # two pieces
            for i, strong_piece in enumerate(pieces):
                for weak_piece in pieces[i:]:
                    yield f'K{strong_piece}{weak_piece}k'
                    yield f'K{strong_piece}k{weak_piece.lower()}'
            # one pawn one piece
            for piece in pieces:
                for rank in range(7, 1, -1):
                    yield f'K{piece}P{rank}k'
                    yield f'K{piece}kp{rank}'
            # two pawns
            for strong_rank in range(7, 1, -1):
                for weak_rank in range(strong_rank, 1, -1):
                    yield f'KP{strong_rank}P{weak_rank}k'
                    yield f'KP{strong_rank}kp{weak_rank}'
        if piece_count >= 5:
            raise NotImplementedError()
            # for i, strong_piece in enumerate(pieces):
            #     for j, middle_piece in enumerate(pieces[i:]):
            #         for weak_piece in pieces[j:]:
            #             yield f'K{strong_piece}{middle_piece}{weak_piece}k'
            #             yield f'K{strong_piece}{middle_piece}k{weak_piece.lower()}'
            #             if values[strong_piece] > values[middle_piece] + values[weak_piece]:
            #                 yield f'K{strong_piece}k{middle_piece.lower()}{weak_piece.lower()}'
            #             else:
            #                 yield f'K{middle_piece}{weak_piece}k{strong_piece.lower()}'
