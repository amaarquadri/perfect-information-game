from src.utils.active_game import ActiveGame as GameClass
from src.move_selection.mcts import AsyncMCTS
from src.heuristics.network import Network
from src.ui.pygame_ui import PygameUI
from src.utils.utils import get_training_path
from time import sleep


def main(user_is_player_1=False):
    pygame_ui = PygameUI(GameClass)
    # pygame_ui.set_user_position()

    # E (0hrs): model_reinforcement-1603784367.0827568
    # M (8hrs): model_reinforcement-1603813336.0729053
    # H (19.5hrs): model_reinforcement-1603854873.4800494
    # network = Network(GameClass, f'{get_training_path(GameClass)}/models/model_best.h5')
    move_chooser = AsyncMCTS(GameClass, pygame_ui.get_position(), time_limit=3, network=None, threads=1)
    move_chooser.start()

    while True:
        if GameClass.is_player_1_turn(pygame_ui.get_position()) == user_is_player_1:
            user_chosen_position = pygame_ui.get_user_move()
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
