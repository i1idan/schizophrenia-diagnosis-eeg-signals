from tensorflow.keras.models import Model
from .fft_custom import FFTCustom
from .transformer import Transformer

models = {"FFTCustom": FFTCustom,
          "Transformer": Transformer}


def load_model(model_name, **kwargs) -> Model:
    return models[model_name](**kwargs).get_model()
