from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Conv2D, MaxPooling2D, Input, Dense, Add
from tensorflow.keras.models import Model
import tensorflow as tf


class FFTCustom:

    def __init__(self, **kwargs):
        self.fft_layer = tf.keras.layers.Lambda(self.image_fft)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    @staticmethod
    def image_fft(x):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.signal.fft2d(tf.cast(x, dtype=tf.complex64))
        return tf.cast(tf.abs(x), dtype=tf.float32)

    def get_model(self) -> Model:
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=(112, 112, 1)))

        model.add(Conv2D(8, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        # model.add(fft_layer)

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(self.fft_layer)

        model.add(Dropout(0.25))
        model.add(Conv2D(64, (2, 2), activation='relu'))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # model = Model(inputs=inputs, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        return model


if __name__ == '__main__':
    model = FFTCustom().get_model()
    model.summary()
