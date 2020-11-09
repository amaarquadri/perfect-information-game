from keras.models import load_model
import tensorflowjs as tfjs
from src.utils.active_game import ActiveGame as GameClass
from src.utils.utils import get_training_path


def main():
    file_name = 'model_best'
    model = load_model(f'{get_training_path(GameClass)}/models/{file_name}.h5')
    tfjs.converters.save_keras_model(model, f'{get_training_path(GameClass)}/models/{file_name}_tfjs/')


if __name__ == '__main__':
    main()

