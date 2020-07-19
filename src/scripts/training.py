import numpy as np
import easygui
from src.utils.active_game import ActiveGame as GameClass
from src.learning.learning import SelfPlayReinforcementLearning, MCTSRolloutGameGenerator
from src.utils.utils import get_training_path


def main():
    trainer = SelfPlayReinforcementLearning(GameClass,
                                            f'{get_training_path(GameClass)}/models/model-reinforcement.h5',
                                            threads_per_section=14, game_batch_count=7, expansions_per_move=1500,
                                            c=np.sqrt(2), d=1)
    # trainer = MCTSRolloutGameGenerator(GameClass, threads=14, expansions_per_move=1500, c=np.sqrt(2))
    trainer.start()
    easygui.msgbox('Click to end training', title=f'{GameClass.__name__} Training', ok_button='End Training')
    print('Ending training')
    trainer.terminate()


if __name__ == '__main__':
    main()
