import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense


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


def test(target='gpu'):
    model = get_model()
    with tf.device(target):
        for i in range(10000):
            arg = np.random.rand(10000, 6, 7, 3)
            print(i)
            model.predict(arg)


if __name__ == '__main__':
    tf.config.experimental.list_physical_devices()
    # tf.debugging.set_log_device_placement(True)
    test()
