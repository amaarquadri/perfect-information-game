from perfect_information_game.utils import get_training_path, ActiveGame as GameClass
from perfect_information_game.move_selection import TablebaseChooser
from perfect_information_game.ui import PygameUI
from time import sleep


def main():
    move_chooser = TablebaseChooser(GameClass)
    pos = GameClass.parse_fen('K7/7B/kN6/8/8/8/8/8 w - - - -')
    pygame_ui = PygameUI(GameClass, starting_position=pos)
    move_chooser.position = pos
    move_chooser.start()

    current_position = pygame_ui.get_position()
    while True:
        print('Left click to make a move, or right click to see the optimal move.')
        choice = pygame_ui.click_left_or_right()
        if choice is None:
            print('User quit game!')
            return
        if choice:  # left click, make custom move
            print('Make a move.')
            user_chosen_position = pygame_ui.get_user_move()
            if user_chosen_position is None:
                print('User quit game!')
                return
            print(GameClass.get_move_notation(current_position, user_chosen_position))
            current_position = user_chosen_position
            move_chooser.report_user_move(user_chosen_position)

            if GameClass.is_over(user_chosen_position):
                break
        else:  # right click, view optimal move
            ai_chosen_positions = move_chooser.choose_move()

            print(GameClass.get_move_notation(current_position, ai_chosen_positions[-1]))
            current_position = ai_chosen_positions[-1]

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
