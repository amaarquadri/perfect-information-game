import json
# noinspection PyUnresolvedReferences
import numpy as np  # used in eval
import easygui
from perfect_information_game.heuristics import train_from_scratch
from perfect_information_game.learning import SelfPlayReinforcementLearning, MCTSRolloutGameGenerator
from perfect_information_game.utils import get_training_path


def parse_float(value):
    if type(value) is float:
        return value
    elif type(value) is str:
        try:
            return float(eval(value))
        except Exception as e:  # need to catch Exception instead of ValueError because of eval
            raise ValueError(f'Cannot interpret {value} as a float!') from e


def launch_trainer(GameClass, trainer):
    trainer.start()

    text = easygui.enterbox('End Training? Timeout (seconds):', title=f'{GameClass.__name__} Training', default='120')
    if text is None:
        print('Timeout not given! Using default of 1 hour.')
        timeout = 3600
    else:
        try:
            timeout = int(text)
        except ValueError:
            print('Timeout invalid! Must be an integer. Using default of 1 hour.')
            timeout = 3600

    print(f'Ending training with timeout of {timeout}')
    trainer.terminate(timeout)


def train_from_config(GameClass, file_name='training_config'):
    with open(f'{get_training_path(GameClass)}/{file_name}.json') as file:
        training_config = json.load(file)

    for training_stage in training_config['stages']:

        if training_stage['training_type'] == 'MCTS Rollout':
            trainer = MCTSRolloutGameGenerator(GameClass, threads=training_stage['threads'],
                                               expansions_per_move=training_stage['expansions_per_move'],
                                               c=parse_float(training_stage['c']))
            launch_trainer(GameClass, trainer)
        elif training_stage['training_type'] == 'Supervised Learning':
            hyper_params = {
                'kernel_size': eval(training_stage['kernel_size']),
                "convolutional_filters_per_layer": training_stage['convolutional_filters_per_layer'],
                "residual_layers": training_stage['residual_layers'],
                "value_head_neurons": training_stage['value_head_neurons'],
                'policy_loss_value': training_stage['policy_loss_value']
            }
            train_from_scratch(GameClass, hyper_params)
        elif training_stage['training_type'] == "Reinforcement Learning":
            hyper_params = training_stage['hyper_params']
            hyper_params['kernel_size'] = eval(hyper_params['kernel_size'])
            trainer = SelfPlayReinforcementLearning(GameClass,
                                                    f'{get_training_path(GameClass)}/models/model_reinforcement.h5',
                                                    threads=training_stage['threads'],
                                                    game_batch_size=training_stage['game_batch_size'],
                                                    expansions_per_move=training_stage['expansions_per_move'],
                                                    c=parse_float(training_stage['c']),
                                                    d=parse_float(training_stage['d']),
                                                    replay_buffer_size=training_stage['replay_buffer_size'])
            launch_trainer(GameClass, trainer)
        else:
            raise ValueError('Unknown training_type: ' + training_stage['training_type'])
