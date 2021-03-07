import unittest
from functools import partial
import numpy as np
from tablebases.tablebase_generator import TablebaseGenerator
from tablebases.tablebase_manager import TablebaseManager
from games.king_of_the_hill_chess import KingOfTheHillChess as GameClass
from utils.utils import OptionalPool


class TestTablebaseGenerator(unittest.TestCase):
    def test_K_vs_k(self):
        generator = TablebaseGenerator(GameClass)
        manager = generator.tablebase_manager
        with OptionalPool(12) as pool:
            generator.generate_tablebase('Kk', pool)

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
        move_board_bytes = GameClass.encode_board_bytes(move)
        return move_board_bytes, terminal_distance

    @staticmethod
    def validate_node(board_bytes, graph, manager):
        move_bytes = graph[board_bytes][0]
        *_, expected_terminal_distance = manager.parse_move_bytes(move_bytes)
        seen_positions = set()
        pos = board_bytes
        depth = 0
        while pos is not None:
            if pos in seen_positions:
                if expected_terminal_distance != np.inf:
                    raise AssertionError(f'Terminal distance is finite but infinite loop found! '
                                         f'Fen: {GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))}')
                break
            seen_positions.add(pos)
            pos = graph[board_bytes][0]
            depth += 1
        else:
            if expected_terminal_distance != depth:
                raise AssertionError(f'Expected terminal distance of {expected_terminal_distance} '
                                     f'but got {depth} based on graph traversal! '
                                     f'Fen: {GameClass.encode_fen(GameClass.parse_board_bytes(board_bytes))}')

    def test_terminal_distances(self):
        descriptor = 'KQkq'
        manager = TablebaseManager(GameClass)
        manager.ensure_loaded(descriptor)
        nodes = manager.tablebases[descriptor]

        with OptionalPool(12) as pool:
            new_values = pool.starmap(partial(self.get_move_board_bytes_and_terminal_distance, manager=manager),
                                      list(nodes.items()))
            graph = {board_bytes: new_values for board_bytes, new_value in zip(nodes.keys(), new_values)}
            print('Created graph')

            pool.map(partial(self.validate_node, graph=graph, manager=manager), graph.keys())
        print('Completed!')


if __name__ == '__main__':
    unittest.main()
