from keras.models import load_model
import tensorflowjs as tfjs
from src.utils.active_game import ActiveGame as GameClass
from src.utils.utils import get_training_path


def main():
    model = load_model(f'{get_training_path(GameClass)}/models/model_best.h5')
    tfjs.converters.save_keras_model(model, f'{get_training_path(GameClass)}/models/tfjs/')


if __name__ == '__main__':
    main()

