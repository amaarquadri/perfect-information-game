from src.games.Amazons import Amazons as Game
from src.move_selection.MCTS import AsyncMCTS
from src.ui.pygame_ui import PygameUI


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
