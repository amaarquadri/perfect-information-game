from src.games.TicTacToe import TicTacToe as Game
from src.move_selection.MiniMax import MiniMax

if __name__ == '__main__':
    board = Game()
    move_chooser = MiniMax(Game, None, 10)

    while not Game.is_over(board.get_state()):
        board.draw()
        print('\n')
        moves = Game.get_possible_moves(board.get_state())
        for i, move in enumerate(moves):
            print(i, ':\n', Game.get_human_readable_representation(move), '\n')

        user_choice = int(input("Pick a Move: "))
        board.set_state(moves[user_choice])

        board.draw()
        print("\n")
        board.set_state(move_chooser.choose_move(board.get_state()))
    print("Result: ", Game.get_winner(board.get_state()))
