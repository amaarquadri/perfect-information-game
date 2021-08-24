from time import time
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from perfect_information_game.move_selection.mcts import RolloutNode
from perfect_information_game.games import Chess as GameClass


def get_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(6, 7, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def benchmark_inference(target='gpu'):
    """
    Use this function as a quick test to verify that Tensorflow is running on the GPU.
    You will see significantly faster performance on the GPU.
    You can also check that a bunch of your VRAM is allocated in task manager.

    In one case, Tensorflow was running on my CPU and tensorflow-cpu was installed in my virtual environment.
    If this happens, recreating the virtual environment might help.

    @param target: Either 'gpu' or 'cpu'
    @return:
    """
    model = get_model()
    with tf.device(target):
        for i in range(10000):
            arg = np.random.rand(10000, 6, 7, 3)
            print(i)
            model.predict(arg)


def benchmark_rollouts(trials=5):
    times = []
    for _ in range(trials):
        start_time = time()
        root = RolloutNode(GameClass.STARTING_STATE, parent=None, GameClass=GameClass)
        while root.count_expansions() < 1000:
            root.choose_expansion_node().expand()
        times.append(time() - start_time)
    print(np.mean(times))
    print(np.std(times))


if __name__ == '__main__':
    tf.config.experimental.list_physical_devices()
    # tf.debugging.set_log_device_placement(True)
    benchmark_inference()
    # benchmark_rollouts()
