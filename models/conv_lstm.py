from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Conv1D, GRU
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model


class ConvLstm:

    def __init__(self, **kwargs):
        self.n_length = 100
        self.n_features = 1

    def get_model(self) -> Model:
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=8, kernel_size=5, activation='relu'),
                                  input_shape=(None, self.n_length, self.n_features)
                                  )
                  )
        model.add(TimeDistributed(Conv1D(filters=4, kernel_size=5, activation='relu'),
                                  input_shape=(None, self.n_length, self.n_features)))
        model.add(TimeDistributed(Conv1D(filters=2, kernel_size=5, activation='relu'),
                                  input_shape=(None, self.n_length, self.n_features)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(512))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


if __name__ == '__main__':
    model = ConvLstm().get_model()
    model.summary()
