from src.games.connect4 import Connect4 as GameClass
from src.move_selection.mcts import AsyncMCTS
from src.heuristics.network import Network
from src.ui.pygame_ui import PygameUI


def main():
    pygame_ui = PygameUI(GameClass)
    # pygame_ui.set_user_position()

    network = Network(GameClass, f'../../training/{GameClass.__name__}/models/model19.h5')
    move_chooser = AsyncMCTS(GameClass, pygame_ui.get_position(), time_limit=5, network=network, threads=1)
    move_chooser.start()

    while True:
        user_chosen_position = pygame_ui.get_user_move()
        if GameClass.is_over(user_chosen_position):
            break

        ai_chosen_position = move_chooser.choose_move(user_chosen_position)
        pygame_ui.draw(ai_chosen_position)
        if GameClass.is_over(ai_chosen_position):
            break
    print('Winner: ', GameClass.get_winner(pygame_ui.get_position()))

    move_chooser.terminate()
    pygame_ui.quit_on_x()


if __name__ == '__main__':
    main()
