import keras

from constants.paths import PARAMS_FILENAMES
from core.file_manager.loadings import load_json


def load_model(model_dir):
    model_name = model_dir.split('/')[-2]
    model_path = f'{model_dir}{model_name}.hdf5'

    return keras.models.load_model(model_path)

def load_params(type, model_dir):
    """

    :param type:            ['data' | 'network' | 'training']
    :param model_dir:
    :return:
    """

    filepath = f'{model_dir}{PARAMS_FILENAMES[type]}'
    params = load_json(filepath)

    return params