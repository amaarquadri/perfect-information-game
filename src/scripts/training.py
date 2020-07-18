import numpy as np
import easygui
from src.games.connect4 import Connect4 as GameClass
from src.learning.learning import SelfPlayReinforcementLearning, MCTSRolloutGameGenerator


def main():
    trainer = SelfPlayReinforcementLearning(GameClass,
                                            f'../heuristics/{GameClass.__name__}/models/model-reinforcement.h5',
                                            threads_per_section=14, game_batch_count=7, expansions_per_move=1500,
                                            c=np.sqrt(2), d=1)
    # trainer = MCTSRolloutGameGenerator(GameClass, threads=14, c=np.sqrt(2))
    trainer.start()
    easygui.msgbox('Click to end training', title=f'{GameClass.__name__} Training', ok_button='End Training')
    print('Ending training')
    trainer.terminate()


def sanitize():
    import os
    import pickle
    from shutil import copyfile
    for file in os.listdir(f'../heuristics/{GameClass.__name__}/games/mcts_network0_games'):
        error = False
        try:
            with open(f'../heuristics/{GameClass.__name__}/games/mcts_network0_games' + '/' + file, 'rb') as fin:
                pickle.load(fin)
        except:
            error = True
        if error:
            pass
            # copyfile(f'../heuristics/{GameClass.__name__}/games/raw_mcts_games/' + file,
            #          f'../heuristics/{GameClass.__name__}/games/raw_mcts_games/dead/' + file)
        else:
            copyfile(f'../heuristics/{GameClass.__name__}/games/mcts_network0_games/' + file,
                     f'../heuristics/{GameClass.__name__}/games/mcts_network0_games/temp/' + file)


if __name__ == '__main__':
    main()
