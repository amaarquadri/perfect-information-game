from .game import Game
import numpy as np
from ..utils.utils import iter_product


class Checkers(Game):
    """
    The game state is represented by an 8x8x6 matrix. The 6 layers correspond to:
    red pieces, red kings, black pieces, black kings, double jump, whose turn.
    Kings can only move a distance of 1, but in any direction.
    Double jump will be all zeros if it is the start of someone's turn, and it will have a single 1 after a player makes
    a move that permits a double jump. In this case, whose turn it is will remain unchanged.

    The second jump of the double jump will be inputted by clicking the piece and its destination as usual.
    If a double jump is possible, then it must be made.
    Double jumps are not possible after a piece becomes a king.
    """

    RED = np.array(5 * [8 * [0]] + [4 * [1, 0]] + [4 * [0, 1]] + [4 * [1, 0]])
    BLACK = np.array([4 * [0, 1]] + [4 * [1, 0]] + [4 * [0, 1]] + 5 * [8 * [0]])
    STARTING_STATE = np.stack([RED, np.zeros((8, 8)),
                               BLACK, np.zeros((8, 8)),
                               np.zeros((8, 8)),
                               np.ones((8, 8))], axis=-1).astype(np.uint8)
    STATE_SHAPE = STARTING_STATE.shape  # 8, 8, 6
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE  # 8, 8, 6
    BOARD_SHAPE = (ROWS, COLUMNS)  # 8, 8
    MOVE_SHAPE = (ROWS, COLUMNS // 2, 4)  # 8, 4, 4
    MOVE_DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    REPRESENTATION_LETTERS = ['r', 'R', 'b', 'B']
    CLICKS_PER_MOVE = 2
    REPRESENTATION_FILES = ['dark_square', 'red_circle_dark_square', 'red_circle_k_dark_square',
                            'black_circle_dark_square', 'black_circle_k_dark_square']

    def __init__(self, state=STARTING_STATE):
        super().__init__(state)

    def perform_user_move(self, clicks):
        (start_i, start_j), (end_i, end_j) = clicks
        is_double_jump = np.any(self.state[:, :, -2] == 1)
        if is_double_jump:
            pass
        new_state = self.null_move(self.state)

        is_king = np.any(self.state[start_i, start_j, [1, 3]] == 1) or \
                  end_j == 0 or end_j == Checkers.ROWS - 1
        new_piece = [0] * 4
        new_piece[2 * (not self.is_player_1_turn(self.state)) + is_king] = 1
        new_state[end_i, end_j, :4] = new_piece
        new_state[start_i, start_j, :4] = [0] * 4
        if np.abs(end_j - start_j) == 2:
            new_state[(start_i + end_i) // 2, (start_j + end_j) // 2, :4] = [0] * 4

        for move in self.get_possible_moves(self.state):
            if np.all(move == new_state):
                break
        else:
            raise ValueError('Invalid Move!')

        self.state = new_state

    @classmethod
    def get_possible_moves(cls, state):
        """
        :return:
        """
        # TODO: include multi captures
        is_double_jump = np.any(state[:, :, -2] == 1)
        moves = []
        if cls.is_player_1_turn(state):
            for i, j, ((di, dj), king_move) in iter_product(cls.BOARD_SHAPE,
                                                            zip(cls.MOVE_DIRECTIONS, [True, True, False, False])):
                if np.any(state[i, j, 1] if king_move else state[i, j, :2]) \
                        and cls.is_valid(i + di, j + dj):
                    if cls.is_empty(state, i + di, j + dj) and not is_double_jump:
                        move = cls.null_move(state)
                        move[i + di, j + dj, :2] = move[i, j, :2] if i + di > 0 else [0, 1]
                        move[i, j, :2] = [0, 0]
                        moves.append(move)
                    elif cls.is_valid(i + 2 * di, j + 2 * dj) \
                            and cls.is_empty(state, i + 2 * di, j + 2 * dj) \
                            and np.any(state[i + di, j + dj, 2:4]):
                        # manually copy the move and apply the capture without switching whose turn it is
                        move = np.copy(state)
                        move[i + 2 * di, j + 2 * dj, :2] = move[i, j, :2] if i + 2 * di > 0 else [0, 1]
                        move[i, j, :2] = [0, 0]
                        move[i + di, j + dj, 2:4] = [0, 0]

                        move = cls.apply_double_jump_rules(state, move, i, j, di, dj)
                        moves.append(move)
        else:
            for i, j, ((di, dj), king_move) in iter_product(cls.BOARD_SHAPE,
                                                            zip(cls.MOVE_DIRECTIONS, [False, False, True, True])):
                if np.any(state[i, j, 3] if king_move else state[i, j, 2:4]) \
                        and cls.is_valid(i + di, j + dj):
                    if cls.is_empty(state, i + di, j + dj) and not is_double_jump:
                        move = cls.null_move(state)
                        move[i + di, j + dj, 2:4] = move[i, j, 2:4] if i + di < 7 else [0, 1]
                        move[i, j, 2:4] = [0, 0]
                        moves.append(move)
                    elif cls.is_valid(i + 2 * di, j + 2 * dj) \
                            and cls.is_empty(state, i + 2 * di, j + 2 * dj) \
                            and np.any(state[i + di, j + dj, :2]):
                        # manually copy the move and apply the capture without switching whose turn it is
                        move = np.copy(state)
                        move[i + 2 * di, j + 2 * dj, 2:4] = move[i, j, 2:4] if i + 2 * di < 7 else [0, 1]
                        move[i, j, 2:4] = [0, 0]
                        move[i + di, j + dj, :2] = [0, 0]

                        move = cls.apply_double_jump_rules(state, move, i, j, di, dj)
                        moves.append(move)
        return moves

    @classmethod
    def apply_double_jump_rules(cls, state, move, i, j, di, dj):
        """
        @param state: The state prior to the move being made.
        @param move: The state after the move has been made, without the turn having switched.
        @param i:
        @param j:
        @param di:
        @param dj:
        """
        # remove any previous double moves
        move[i, j, -2] = 0

        became_king = np.none(state[i, j, [1, 3]] == 1) and \
            np.any(move[i + 2 * di, j + 2 * dj, [1, 3]] == 1)

        if not became_king and cls.can_double_jump(move, i + 2 * di, j + 2 * dj):
            # set the double jump to 1, and don't change whose turn it is
            move[i + 2 * di, j + 2 * dj, -2] = 1
        else:
            # switch whose turn it is
            move = cls.null_move(move)
        return move

    @classmethod
    def can_double_jump(cls, state, i, j):
        is_king = np.any(state[i, j, [1, 3]])
        if cls.is_player_1_turn(state):
            for (di, dj), king_move in zip(cls.MOVE_DIRECTIONS, [True, True, False, False]):
                if (is_king or not king_move) and \
                        cls.is_valid(i + 2 * di, j + 2 * dj) and cls.is_empty(state, i + 2 * di, j + 2 * dj) and \
                        np.any(state[i + di, j + dj, 2:4]):
                    return True
        else:
            for (di, dj), king_move in zip(cls.MOVE_DIRECTIONS, [False, False, True, True]):
                if (is_king or not king_move) and \
                        cls.is_valid(i + 2 * di, j + 2 * dj) and cls.is_empty(state, i + 2 * di, j + 2 * dj) and \
                        np.any(state[i + di, j + dj, :2]):
                    return True
        return False

    @classmethod
    def get_legal_moves(cls, state):
        raise NotImplementedError()

    @classmethod
    def is_over(cls, state):
        return len(cls.get_possible_moves(state)) == 0

    @classmethod
    def get_winner(cls, state):
        if not cls.is_over(state):
            raise Exception('Game is not over!')
        if cls.is_player_1_turn(state):
            return -1
        else:
            return 1

    @classmethod
    def is_empty(cls, state, i, j):
        return np.all(state[i, j, :-1] == 0)

    @classmethod
    def null_move(cls, state):
        move = super(Checkers, cls).null_move(state)
        move[:, :, -2] = 0  # remove double jump possibilities
        return move
