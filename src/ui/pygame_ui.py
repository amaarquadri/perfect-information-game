from ..utils.utils import iter_product
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame  # noqa: E402


class PygameUI:
    """
    The highlight image must have a transparent background (https://onlinepngtools.com/create-transparent-png).
    """
    HIGHLIGHT_IMAGE_PATH = '../../resources/blue_border.png'
    FLIP_LR = False

    def __init__(self, GameClass, starting_position=None):
        self.GameClass = GameClass
        pygame.init()
        # Note pygame inverts x and y
        self.canvas = pygame.display.set_mode((64 * GameClass.COLUMNS, 64 * GameClass.ROWS))
        self.imgs = [pygame.image.load(f'../../resources/{file_name}.png')
                     for file_name in GameClass.REPRESENTATION_FILES]
        self.highlight_img = pygame.image.load(PygameUI.HIGHLIGHT_IMAGE_PATH)
        if starting_position is None:
            starting_position = GameClass.STARTING_STATE
        self.board = GameClass(starting_position)
        self.last_indices = GameClass.get_img_index_representation(starting_position)
        self.draw()

    def reset_board(self):
        self.draw(self.GameClass.STARTING_STATE)

    def get_position(self):
        return self.board.get_state()

    def draw(self, position=None):
        """
        Draws the given position to the screen. If no position is given, then the current position will be drawn.
        """
        if position is not None:
            self.board.set_state(position)
        else:
            position = self.board.get_state()

        indices = self.GameClass.get_img_index_representation(position)
        changed_indices = indices != self.last_indices
        self.last_indices = indices

        for i, j in iter_product(self.GameClass.BOARD_SHAPE):
            img = self.imgs[indices[i, j]]
            if PygameUI.FLIP_LR:
                x = 64 * (self.GameClass.COLUMNS - 1 - i)
            else:
                x = 64 * i
            y = 64 * j

            self.canvas.blit(img, (y, x))  # Note pygame inverts x and y

            # noinspection PyUnresolvedReferences
            if changed_indices[i, j]:
                self.canvas.blit(self.highlight_img, (y, x))

        pygame.display.flip()
        # Not sure why this is necessary to get the screen to update
        self.flush()

    def get_user_move(self):
        """
        Get user input to perform the next move.
        The user performs a number of clicks specified by GameClass.CLICKS_PER_MOVE.
        If the move is invalid, the user is prompted to re-enter a valid move recursively.

        :return: The resulting board state after the user's move, or None if the user quit the game.
        """
        self.flush()

        clicks = []
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos[0] // 64, event.pos[1] // 64
                    if PygameUI.FLIP_LR:
                        x = self.GameClass.COLUMNS - 1 - x

                    clicks.append((y, x))  # Note pygame inverts x and y

                    if len(clicks) == self.GameClass.CLICKS_PER_MOVE:
                        try:
                            self.board.perform_user_move(clicks)
                            self.draw()
                            return self.board.get_state()
                        except ValueError:
                            print('Invalid Move! Try again.')
                            clicks = []

    def set_user_position(self):
        """
        Allows the user to make an arbitrary number of moves from the current position.
        Each move consists of GameClass.CLICKS_PER_MOVE left clicks.
        Anytime an invalid move is entered, the user is notified and prompted to try again.
        The function terminates once the user right clicks to indicate that they have made as many moves as desired.

        :return: True if the user made any number of moves (including 0) and then clicked the right button,
                 and False if the user quit pygame.
        """
        self.flush()

        clicks = []
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_LEFT:
                        x, y = event.pos[0] // 64, event.pos[1] // 64
                        if PygameUI.FLIP_LR:
                            x = self.GameClass.COLUMNS - 1 - x
                        clicks.append((y, x))  # Note pygame inverts x and y
                        if len(clicks) == self.GameClass.CLICKS_PER_MOVE:
                            try:
                                self.board.perform_user_move(clicks)
                                self.draw()
                            except ValueError:
                                print('Invalid Move! Try again.')
                            clicks = []
                    elif event.button == pygame.BUTTON_RIGHT:
                        return True

    def show_game(self, positions, starting_index=0, messages=None):
        """
        Shows the given sequence of positions to the user. The user can right and left click to navigate through them.
        This function blocks until the user decides to close the program via the X button.
        This function blocks until the user decides to close the program via the X button.

        :param positions: The list of positions to show.
        :param starting_index: The index in the list of positions to start at. Negative indices are supported.
        :param messages: An optional list of messages to print with each position.
                         If provided, it must be the same length as positions.
        """
        if messages is not None and len(messages) != len(positions):
            raise Exception('Length of messages must match length of positions!')
        if not -len(positions) <= starting_index < len(positions):
            raise Exception('starting_index out of bounds!')

        i = starting_index if starting_index >= 0 else len(positions) + starting_index
        while True:
            val = self.click_left_or_right()
            if val is None:
                return
            if val:
                i = min(i + 1, len(positions) - 1)
            else:
                i = max(i - 1, 0)

            self.draw(positions[i])
            if messages is not None:
                print(messages[i])

    @staticmethod
    def flush():
        """
        Clears all pending mouse clicks events from pygame's queue.
        This is useful for removing queued up clicks before asking the user for a new click.
        This is also useful for preventing a 'Not Responding' dialog.
        """
        pygame.event.clear(pygame.MOUSEBUTTONDOWN)

    @staticmethod
    def quit_on_x():
        """
        Waits for the user to quit.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    @staticmethod
    def click_to_continue():
        """
        :return: True if the user clicks, and False if the user quits.
        """
        PygameUI.flush()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return True

    @staticmethod
    def click_left_or_right():
        """
        :return: True if the user left clicks, False if the user right clicks, and None if the user quits.
        """
        PygameUI.flush()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_LEFT:
                        return True
                    if event.button == pygame.BUTTON_RIGHT:
                        return False
