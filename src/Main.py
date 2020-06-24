from src.games.Amazons import Amazons as Game
from src.move_selection.MCTS import AsyncMCTS

if __name__ == '__main__':
    board = Game()
    move_chooser = AsyncMCTS(Game, board.get_state())
    move_chooser.start()
    time_limit = 3

    while not Game.is_over(board.get_state()):
        board.draw(move_prompt=True)
        print('\n')

        user_choice = input("Pick a Move: ")
        board.perform_user_move(user_choice)

        board.draw()
        print("\n")
        board.set_state(move_chooser.choose_move(board.get_state()))
    move_chooser.terminate()
    print("Result: ", Game.get_winner(board.get_state()))
