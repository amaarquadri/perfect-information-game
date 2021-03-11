# Use Gan: one network choosing placement, one shooting
# generator network maps a vector from a random state space to the output choice of piece placement
import numpy as np
from games.game import Game
from utils.utils import iter_product


class Battleship(Game):
    """
    Battleship works fairly differently from other games because the game is disjoint.
    The actions of 1 player have no bearing on the strategy of the other player.
    Thus, the machine learning aspect will only consider one half of the game when training and playing.

    Default Rules:
        Board size: 10x10
        Ship sizes: 2, 3, 3, 4, 5

    Game State:
        10x10x7, where the 7 channels correspond to: p1 pieces, p1 hits, p1 misses, p2 pieces, p2 hits, p2 misses, turn

    Placement Network is not feasible:
        Goal would be to maps a random vector (from latent space) to a placement vector with shape representing all
        possible moves: 2x10x9x2x10x8x2x10x8x2x10x7x2x10x6. This is a 77.4144 billion dimensional output space.
        Of this 30,093,975,536 moves are legal (due to no ships overlapping) which is about 39%.
        Even reducing by a factor of 8 due to symmetry leaves 3,761,746,942 unique legal moves.
        Thus this is not feasible, so starting positions will be chosen randomly instead.

    Policy Network:
        Takes part of the board state as input (only takes the hits and misses for the player it is considering).
        Policy: Generates a probability distribution over all possible moves, similar to for other games.
        No value network.
    """
    W = 10
    STARTING_STATE = None
    SHIP_LENGTHS = [2, 3, 3, 4, 5]
    STATE_SHAPE = (W, W, 7)
    ROWS, COLUMNS, FEATURE_COUNT = STATE_SHAPE
    BOARD_SHAPE = (ROWS, COLUMNS)
    MOVE_SHAPE = BOARD_SHAPE
    REPRESENTATION_LETTERS = ['r', 'w']
    REPRESENTATION_FILES = ['dark_square', 'red_circle_dark_square', 'white_circle_dark_square']
    CLICKS_PER_MOVE = 1

    def __init__(self, player_1_pieces=None, player_2_pieces=None):
        if player_1_pieces is None:
            player_1_pieces = Battleship.get_random_starting_position()
        if player_2_pieces is None:
            player_2_pieces = Battleship.get_random_starting_position()

        starting_state = np.stack(player_1_pieces, np.zeros_like(player_1_pieces), np.zeros_like(player_1_pieces),
                                  player_2_pieces, np.zeros_like(player_2_pieces), np.zeros_like(player_2_pieces),
                                  np.ones_like(player_1_pieces))
        super().__init__(starting_state)

    @classmethod
    def get_ai_state(cls, state):
        """
        :param state:
        :return: A (W, W, 2) subset of the state corresponding to the hits and misses for the player whose turn it is.
        """
        if cls.is_player_1_turn(state):
            return state[:, :, 1:3]
        else:
            return state[:, :, 4:6]

    @staticmethod
    def get_random_starting_position():
        starting_position = np.zeros(Battleship.BOARD_SHAPE)

        for ship_length in Battleship.SHIP_LENGTHS:
            piece_position = np.zeros(Battleship.BOARD_SHAPE)

            if np.random.random_sample() < 0.5:
                # horizontal piece placement case
                row = np.random.randint(0, Battleship.W + 1)
                column = np.random.randint(0, Battleship.W - ship_length + 2)
                piece_position[row, column:column + ship_length] = 1
            else:
                # vertical piece placement case
                row = np.random.randint(0, Battleship.W - ship_length + 2)
                column = np.random.randint(0, Battleship.W + 1)
                piece_position[row:row + ship_length, column] = 1

            if np.any(np.logical_and(starting_position, piece_position)):
                # Failed due to overlap, recurse and try again
                # Note that failure will only happen about 61% of the time, so this is a fairly sensible implementation
                # In theory this could result in a stack overflow, but the chance of more than 100 failures is less than
                # 1 in 3.9 * 10 ^ 21
                return Battleship.get_random_starting_position()

            starting_position = np.logical_or(starting_position, piece_position)

        return starting_position

    def perform_user_move(self, clicks):
        i, j = clicks[0]

        if self.is_player_1_turn(self.state):
            hit_miss_slice = slice(1, 3)
            opponent_pieces_slice = 3
        else:
            hit_miss_slice = slice(4, 6)
            opponent_pieces_slice = 0

        if np.any(self.state[i, j, hit_miss_slice] == 1):
            raise ValueError('You have already shot that spot!')

        new_state = self.null_move(self.state)
        new_state[i, j, hit_miss_slice] = [1, 0] if new_state[i, j, opponent_pieces_slice] else [0, 1]
        self.state = new_state

    @classmethod
    def get_possible_moves(cls, state):
        moves = []
        if cls.is_player_1_turn(state):
            hit_miss_slice = slice(1, 3)
            opponent_pieces_slice = 3
        else:
            hit_miss_slice = slice(4, 6)
            opponent_pieces_slice = 0

        for i, j in iter_product(Battleship.BOARD_SHAPE):
            if np.all(state[i, j, hit_miss_slice] == 0):
                move = cls.null_move(state)
                move[i, j, hit_miss_slice] = [1, 0] if move[i, j, opponent_pieces_slice] else [0, 1]
                moves.append(move)
        return moves

    @classmethod
    def get_legal_moves(cls, state):
        if cls.is_player_1_turn(state):
            return np.logical_or(state[:, :, 1], state[:, :, 2]) == 0
        else:
            return np.logical_or(state[:, :, 4], state[:, :, 5]) == 0

    @classmethod
    def is_over(cls, state):
        return np.logical_and(state[:, :, 3], state[:, :, 1]) == state[:, :, 3] or \
               np.logical_and(state[:, :, 0], state[:, :, 4]) == state[:, :, 0]

    @classmethod
    def get_winner(cls, state):
        if np.logical_and(state[:, :, 3], state[:, :, 1]) == state[:, :, 3]:
            return 1
        if np.logical_and(state[:, :, 0], state[:, :, 4]) == state[:, :, 0]:
            return -1
        raise ValueError('Game is not finished!')
