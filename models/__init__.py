from tensorflow.keras.models import Model
from .custom_fft import FFTCustom

models = {"FFTCustom": FFTCustom}


def load_model(model_name, **kwargs) -> Model:
    return models[model_name](**kwargs).get_model()
