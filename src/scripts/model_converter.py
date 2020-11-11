from keras.models import load_model
import tensorflowjs as tfjs
from src.utils.active_game import ActiveGame as GameClass
from src.utils.utils import get_training_path
import os
import fileinput


def main():
    for difficulty in ['easy', 'medium', 'hard']:
        output_folder = f'{get_training_path(GameClass)}/models/tfjs_models'
        output_json = f'{output_folder}/othello_{difficulty}_model.json'
        output_weights_file_name = f'othello_{difficulty}_weights'

        model = load_model(f'{get_training_path(GameClass)}/models/model_{difficulty}.h5')
        tfjs.converters.save_keras_model(model, output_folder)

        os.rename(f'{output_folder}/model.json', output_json)
        os.rename(f'{output_folder}/group1-shard1of1.bin', f'{output_folder}/{output_weights_file_name}.bin')

        with fileinput.FileInput(output_json, inplace=True) as file:
            for line in file:
                print(line.replace('group1-shard1of1', output_weights_file_name), end='')


if __name__ == '__main__':
    main()

