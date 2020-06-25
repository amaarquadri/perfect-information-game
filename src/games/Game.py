from abc import ABC, abstractmethod
import numpy as np


class Game(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_state(self):
        """
        The result will be a matrix with shape (n, m, k) where the board has dimensions of nxm and
        k is the number of features.
        For example, in checkers n=m=8 and k=5 where the features are
        [red pieces, red kings, black pieces, black kings, is white to move]
        In chess (ignoring 50 move rule and threefold repetition) n=m=8, k=14 and features=[white pawns, white knights,
        white bishops, white rooks, white queens, white kings, black pawns, black knights, black bishops, black rooks,
        black queens, black kings, castling and en passant booleans (on their respective squares), is white to move].


        :return: A numpy matrix representation of the current state of this Game.
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """
        Set the state of this Game.

        :param state: The state in the format specified by get_ML_representation
        """
        pass

    def perform_user_move(self, move):
        """
        Performs the move specified by the string move, on the specified state and returns the resulting state.
        """
        raise NotImplementedError()

    def reset_game(self):
        self.set_state(self.get_starting_state())

    @abstractmethod
    def draw(self, canvas, move_prompt=False):
        """
        The board should be drawn in book-reading fashion. i.e. The first index represents the row from top to bottom
        and the second index represents the column from left to right.
        If canvas is None, then the class will print a representation to the screen.
        If move_prompt is True, then additional information specified by get_human_move_prompt will be shown.
        This allows the user to more easily input a string which can be interpreted as a move using perform_user_move

        :param canvas:
        :param move_prompt: Whether or not to show additional data for prompting the user to make a move.
        """
        pass

    @classmethod
    @abstractmethod
    def get_representation_shape(cls):
        """
        :return: The shape of the result of get_ML_representation
        """
        pass

    @classmethod
    @abstractmethod
    def get_human_readable_representation(cls, state):
        """
        The result will be a matrix with shape (n, m) and dtype=str. Each element will be a single character which
        aggregates as much information as possible from the features.

        :return: A human readable numpy matrix representation of the current state of this Game.
        """
        pass

    @classmethod
    @abstractmethod
    def get_img_index_representation(cls, state):
        """
        The result will be a matrix with shape (n, m) and dtype=int. Each element will be an integer corresponding to
        which image to use to represent that square. The mapping from indices to file names should be provided in a
        class-level constant list called REPRESENTATION_FILES.

        :return: A numpy matrix indicating which images to use for each square in the grid.
        """
        pass

    @classmethod
    @abstractmethod
    def get_starting_state(cls):
        pass

    @classmethod
    @abstractmethod
    def get_possible_moves(cls, state):
        """
        Each resulting board state will be in the form specified by get_ML_representation

        :return: A list of all possible board states that could result from the given state.
        """
        pass

    @classmethod
    def is_player_1_turn(cls, state):
        return np.all(state[:, :, -1])

    @classmethod
    @abstractmethod
    def is_over(cls, state):
        pass

    @classmethod
    @abstractmethod
    def get_winner(cls, state):
        """
        :return: 1 if player 1 won, 0 if draw, -1 if player 2 won.
        """
        pass

    @classmethod
    def null_move(cls, state):
        move = np.copy(state)
        move[:, :, -1] = np.zeros_like(state[:, :, -1]) if cls.is_player_1_turn(state) \
            else np.ones_like(state[:, :, -1])
        return move
