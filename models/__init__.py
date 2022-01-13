from tensorflow.keras.models import Model
from .wavelet_custom import WaveletCustom
from tensorflow.keras.models import Model
from .fft_custom import FFTCustom


models = {"WaveletCustom": WaveletCustom,
          "FFTCustom": FFTCustom}


def load_model(model_name, **kwargs) -> Model:
    return models[model_name](**kwargs).get_model()
