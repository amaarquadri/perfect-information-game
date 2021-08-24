from time import time
import numpy as np
from perfect_information_game.utils import get_training_path
from perfect_information_game.games import Chess as GameClass
from perfect_information_game.ui import PygameUI
from perfect_information_game.heuristics import Network
from perfect_information_game.move_selection.mcts import AsyncMCTS
from perfect_information_game.move_selection import RawNetwork


def play_games(network1, network2, count=1000):
    net1_wins, net2_wins, draws = 0, 0, 0
    for i in range(count):
        position = GameClass.STARTING_STATE
        # net1_evals = []
        # net2_evals = []
        while not GameClass.is_over(position):
            position = network1.choose_move(position)
            # net1_evals.append(network1.evaluation(position))
            if GameClass.is_over(position):
                break

            position = network2.choose_move(position)
            # net2_evals.append(network2.evaluation(position))

        result = GameClass.get_winner(position)
        # net1_rms = np.mean((np.array(net1_evals) - result) ** 2) ** 0.5
        # net2_rms = np.mean((np.array(net2_evals) - result) ** 2) ** 0.5
        # print(f'Winner {i}: {result}')  # , net 1 rms: {net1_rms}, net 2 rms: {net2_rms}')
        print(f'Player 1 wins {net1_wins}, player 2 wins {net2_wins}, draws {draws}')
        if result == 1:
            net1_wins += 1
        elif result == 0:
            draws += 1
        elif result == -1:
            net2_wins += 1
        else:
            raise Exception()
    print(f'Player 1 wins {net1_wins}, player 2 wins {net2_wins}, draws {draws}')


def play_game_with_ui(pygame_ui, move_chooser1, move_chooser2):
    move_chooser1.start()
    move_chooser2.start()

    start_time = time()
    position = pygame_ui.get_position()
    board_states = [position]

    while not GameClass.is_over(position):
        position = move_chooser1.choose_move(position)
        board_states.append(position)
        pygame_ui.flush()
        pygame_ui.draw(position)
        if GameClass.is_over(position):
            break

        position = move_chooser2.choose_move(position)
        pygame_ui.flush()
        pygame_ui.draw(position)
        board_states.append(position)

    print('Winner: ', GameClass.get_winner(position))
    pygame_ui.show_game(board_states, starting_index=0 if time() - start_time < 15 else -1)


def main():
    # pygame_ui = PygameUI(GameClass)
    network1 = Network(GameClass, f'{get_training_path(GameClass)}/models/model_best.h5')
    move_chooser1 = RawNetwork(network1, delay=0)
    # move_chooser1 = AsyncMCTS(GameClass, GameClass.STARTING_STATE, time_limit=3, network=network1)

    network2 = Network(GameClass, f'{get_training_path(GameClass)}/models/model_reinforcement.h5')
    move_chooser2 = RawNetwork(network2, delay=0)
    # move_chooser2 = AsyncMCTS(GameClass, GameClass.STARTING_STATE, time_limit=3, network=network2)

    # play_game_with_ui(GameClass, pygame_ui, move_chooser1, move_chooser2)
    play_games(move_chooser1, move_chooser2)


if __name__ == '__main__':
    main()
