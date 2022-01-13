from tensorflow.keras.models import Model
from .wavelet_custom import WaveletCustom

models = {"WaveletCustom": WaveletCustom}


def load_model(model_name, **kwargs) -> Model:
    return models[model_name](**kwargs).get_model()
