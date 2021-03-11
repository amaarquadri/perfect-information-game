import unittest
from functools import partial
import numpy as np
from tablebases.tablebase_generator import TablebaseGenerator
from tablebases.tablebase_manager import TablebaseManager
from tablebases.symmetry_transform import SymmetryTransform
from games.king_of_the_hill_chess import KingOfTheHillChess as GameClass
from utils.utils import OptionalPool


class TestTablebaseGenerator(unittest.TestCase):
    def test_K_vs_k(self):
        manager = TablebaseManager(GameClass)
        manager.ensure_loaded('Kk')
        nodes = manager.tablebases['Kk']
        if any([manager.parse_move_bytes(move_bytes)[0] == 0 for move_bytes in nodes.values()]):
            raise AssertionError('Draw found in king versus king for king of the hill chess!')

    @staticmethod
    def get_move_board_bytes_and_terminal_distance(board_bytes, move_bytes, manager):
        state = GameClass.parse_board_bytes(board_bytes)
        outcome, start_i, start_j, end_i, end_j, terminal_distance = manager.parse_move_bytes(move_bytes)

        moves = GameClass.get_possible_moves(state)
        if GameClass.is_over(state, moves):
            if terminal_distance != 0:
                raise AssertionError(f'Game is over but terminal_distance != 0. Fen: {GameClass.encode_fen(state)}')
            return None, 0

        move = GameClass.apply_from_to_move(state, start_i, start_j, end_i, end_j)
        move = SymmetryTransform(GameClass, move).transform_state(move)
        move_board_bytes = GameClass.encode_board_bytes(move)
        return move_board_bytes, terminal_distance

    @staticmethod
    def validate_node(board_bytes, graph):
        expected_terminal_distance = graph[board_bytes][1]
        seen_positions = set()
        pos_board_bytes = board_bytes
        depth = 0
        while pos_board_bytes is not None:
            if pos_board_bytes in seen_positions:
                if expected_terminal_distance != np.inf:
                    raise AssertionError(f'Terminal distance is finite but infinite loop found! '
                                         f'Fen: {GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))}')
                break
            seen_positions.add(pos_board_bytes)
            pos_board_bytes = graph[pos_board_bytes][0]
            depth += 1
        else:
            if expected_terminal_distance != depth:
                raise AssertionError(f'Expected terminal distance of {expected_terminal_distance} '
                                     f'but got {depth} based on graph traversal! '
                                     f'Fen: {GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))}')

    def test_terminal_distances(self):
        # passed for:
        # TWO_MAN = 'Kk'
        # THREE_MAN = 'KQk,KRk,KBk,KNk,KPk'
        # FOUR_MAN_NO_ENEMY_NO_DUPLICATE = 'KQRk,KQBk,KQNk,KRBk,KRNk,KBNk'
        # KQkq
        FOUR_MAN_WITH_ENEMY = 'KQkq,KQkr,KQkb,KQkn,KQkp,KRkr,KRkb,KRkn,KRkp,KBkb,KBkn,KBkp,KNkn,KNkp,KPkp'
        manager = TablebaseManager(GameClass)
        for descriptor in FOUR_MAN_WITH_ENEMY.split(',')[1:-6]:
            print(f'Testing {descriptor}')
            manager.ensure_loaded(descriptor)
            nodes = manager.tablebases[descriptor]
            print(f'Loaded nodes for {descriptor}')

            with OptionalPool(12) as pool:
                new_values = pool.starmap(partial(self.get_move_board_bytes_and_terminal_distance, manager=manager),
                                          list(nodes.items()))
                graph = {board_bytes: new_values for board_bytes, new_value in zip(nodes.keys(), new_values)}
                print(f'Created graph for {descriptor}')

                map(partial(self.validate_node, graph=graph), graph.keys())
            print(f'Completed {descriptor}!')


if __name__ == '__main__':
    unittest.main()
