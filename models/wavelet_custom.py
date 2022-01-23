from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Conv2D, MaxPooling2D, Input, Dense, Add
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2


class WaveletCustom:

    def __init__(self, **kwargs):
        self.gabor_layer = tf.keras.layers.Lambda(self.gabor_filter)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    @staticmethod
    def gabor_filter(x):
        x = tf.cast(x, dtype=tf.float32)
        # x = tf.image.rgb_to_grayscale(x)
        params = {'ksize': (3, 3), 'sigma': 1.0, 'theta': 0, 'lambd': 5.0, 'gamma': 0.02}
        kernel = cv2.getGaborKernel(**params)
        kernel = tf.expand_dims(kernel, 2)
        kernel = tf.expand_dims(kernel, 3)
        kernel = tf.cast(kernel, dtype=tf.float32)
        return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')

    def get_model(self) -> Model:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(112, 112, 1)))
        model.add(self.gabor_layer )
        model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu'))
        model.add(Conv2D(8, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        return model
