from .wavelet_custom import WaveletCustom
from tensorflow.keras.models import Model
from .fft_custom import FFTCustom
from .transformer import Transformer
from .conv_lstm import ConvLstm

models = {"WaveletCustom": WaveletCustom,
          "FFTCustom": FFTCustom,
          "Transformer": Transformer,
          "conv_lstm": ConvLstm}


def load_model(model_name, **kwargs) -> Model:
    return models[model_name](**kwargs).get_model()
