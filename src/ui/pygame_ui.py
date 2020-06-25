from src.games.Connect4 import Connect4 as Game
from src.move_selection.MCTS import AsyncMCTS
import pygame
from src.utils.Utils import iter_product


def get_user_clicks(count=1):
    clicks = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                clicks.append((y // 64, x // 64))  # Note pygame inverst x and y
                if len(clicks) == count:
                    return clicks


def quit_on_x():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


def main():
    pygame.init()
    canvas = pygame.display.set_mode((64 * Game.COLUMNS, 64 * Game.ROWS))  # Note pygame inverst x and y
    empty_square = pygame.image.load('../../resources/dark_square.png').convert()
    red_circle = pygame.image.load('../../resources/red_circle_dark_square.png').convert()
    white_circle = pygame.image.load('../../resources/white_circle_dark_square.png').convert()
    black_circle = pygame.image.load('../../resources/black_circle_dark_square.png').convert()
    yellow_circle = pygame.image.load('../../resources/yellow_circle_dark_square.png').convert()

    board = Game()
    move_chooser = AsyncMCTS(Game, board.get_state(), time_limit=1)
    move_chooser.start()

    def draw_board():
        chars = Game.get_human_readable_representation(board.get_state())
        for i, j in iter_product(Game.BOARD_SHAPE):
            if chars[i, j] == 'r':
                img = red_circle
            elif chars[i, j] == 'y':
                img = yellow_circle
            else:
                img = empty_square
            # if chars[i, j] == 'W':
            #     img = white_circle
            # elif chars[i, j] == 'B':
            #     img = black_circle
            # elif chars[i, j] == 'X':
            #     img = red_circle
            # else:
            #     img = empty_square
            canvas.blit(img, (64 * j, 64 * i))  # Note pygame inverst x and y
        pygame.display.flip()

    draw_board()
    while True:
        # get user move
        clicks = get_user_clicks(count=Game.CLICKS_PER_MOVE)
        if clicks is None:
            move_chooser.terminate()
            pygame.quit()
            return
        try:
            board.perform_user_move(clicks)
        except ValueError:
            print('Illegal Move! Try again.')
            continue

        draw_board()
        if Game.is_over(board.get_state()):
            break

        # ai move
        board.set_state(move_chooser.choose_move(board.get_state()))
        draw_board()
        if Game.is_over(board.get_state()):
            break
    print('Winner: ', Game.get_winner(board.get_state()))

    move_chooser.terminate()
    quit_on_x()


if __name__ == '__main__':
    main()
