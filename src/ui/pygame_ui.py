from src.games.TicTacToe import TicTacToe as Game
from src.move_selection.MCTS import AsyncMCTS
import pygame
from src.utils.Utils import iter_product


class PygameUI:
    def __init__(self, GameClass, starting_position=None):
        self.GameClass = GameClass
        pygame.init()
        # Note pygame inverts x and y
        self.canvas = pygame.display.set_mode((64 * GameClass.COLUMNS, 64 * GameClass.ROWS))
        self.imgs = [pygame.image.load(f'../../resources/{file_name}.png')
                     for file_name in GameClass.REPRESENTATION_FILES]
        if starting_position is None:
            starting_position = GameClass.STARTING_STATE
        self.board = GameClass(starting_position)
        self.draw()

    def get_position(self):
        return self.board.get_state()

    def draw(self, position=None):
        if position is not None:
            self.board.set_state(position)
        else:
            position = self.board.get_state()

        indices = self.GameClass.get_img_index_representation(position)
        for i, j in iter_product(self.GameClass.BOARD_SHAPE):
            img = self.imgs[indices[i, j]]
            self.canvas.blit(img, (64 * j, 64 * i))  # Note pygame inverts x and y
        pygame.display.flip()

    def get_user_move(self):
        pygame.event.clear(pygame.MOUSEBUTTONDOWN)

        clicks = []
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    clicks.append((y // 64, x // 64))  # Note pygame inverst x and y
                    if len(clicks) == self.GameClass.CLICKS_PER_MOVE:
                        try:
                            self.board.perform_user_move(clicks)
                            self.draw()
                            return self.board.get_state()
                        except ValueError:
                            print('Invalid Move! Try again.')
                            return self.get_user_move()  # recurse

    @staticmethod
    def quit_on_x():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    @staticmethod
    def click_to_continue():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return True

    @staticmethod
    def click_left_or_right():
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


def main():
    pygame_ui = PygameUI(Game)
    move_chooser = AsyncMCTS(Game, pygame_ui.get_position(), time_limit=3, threads=1)
    move_chooser.start()

    while True:
        user_chosen_position = pygame_ui.get_user_move()
        if Game.is_over(user_chosen_position):
            break

        ai_chosen_position = move_chooser.choose_move(user_chosen_position)
        pygame_ui.draw(ai_chosen_position)
        if Game.is_over(ai_chosen_position):
            break
    print('Winner: ', Game.get_winner(pygame_ui.get_position()))

    move_chooser.terminate()
    pygame_ui.quit_on_x()


if __name__ == '__main__':
    main()
