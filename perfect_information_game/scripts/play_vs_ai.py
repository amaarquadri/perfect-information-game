from perfect_information_game.utils import get_training_path
from perfect_information_game.games import Chess as GameClass
from perfect_information_game.move_selection import RawNetwork
from perfect_information_game.move_selection.mcts import AsyncMCTS
from perfect_information_game.heuristics import Network
from perfect_information_game.ui import PygameUI
from time import sleep


def main(user_is_player_1=False):
    pygame_ui = PygameUI(GameClass)
    # pygame_ui.set_user_position()

    # network = Network(GameClass, f'{get_training_path(GameClass)}/models/model_hard.h5')
    move_chooser = AsyncMCTS(GameClass, pygame_ui.get_position(), time_limit=3, network=None, threads=1)
    move_chooser.start()

    while True:
        if GameClass.is_player_1_turn(pygame_ui.get_position()) == user_is_player_1:
            user_chosen_position = pygame_ui.get_user_move()
            if user_chosen_position is None:
                print('User quit game!')
                return
            move_chooser.report_user_move(user_chosen_position)
            if GameClass.is_over(user_chosen_position):
                break
        else:
            ai_chosen_positions = move_chooser.choose_move()
            pygame_ui.draw(ai_chosen_positions[0])
            for subsequent_position in ai_chosen_positions[1:]:
                sleep(1)
                pygame_ui.draw(subsequent_position)
            if GameClass.is_over(ai_chosen_positions[-1]):
                break
    print('Winner: ', GameClass.get_winner(pygame_ui.get_position()))

    move_chooser.terminate()
    pygame_ui.quit_on_x()


if __name__ == '__main__':
    main()
