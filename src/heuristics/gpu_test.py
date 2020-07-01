import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv3D, Flatten, Dense


def get_model():
    model = Sequential()
    # model.add(Conv3D(32, (3, 3, 1), input_shape=(6, 7, 3, 1), activation='relu'))
    # model.add(Conv3D(32, (3, 3, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def test_gpu():
    model = get_model()
    arg = np.random.rand(1000_000, 6, 7, 3, 1)
    with tf.device('/GPU:0'):
        for i in range(10):
            print(i)
            model.predict(arg)


if __name__ == '__main__':
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()

    # tf.config.experimental.list_physical_devices()
    # tf.debugging.set_log_device_placement(True)
    # test_gpu()
