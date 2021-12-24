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

    @staticmethod
    def conv_block(x):
        x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1),
                   activation='relu')(x)

        x = Conv2D(8, kernel_size=(2, 2), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(16, (2, 2), activation='relu')(x)
        x = Conv2D(16, (2, 2), activation='relu')(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (2, 2), activation='relu')(x)
        x = Conv2D(32, (2, 2), activation='relu')(x)

        x = Dropout(0.5)(x)
        x = Conv2D(64, (2, 2), activation='relu')(x)

        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        return x

    def get_model(self) -> Model:
        inputs = Input(shape=(112, 112, 1))
        x1 = self.fft_layer(inputs)
        x1 = self.conv_block(x1)
        x2 = self.conv_block(inputs)
        x = Add()([x1, x2])
        # x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        return model
