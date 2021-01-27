from core.file_manager.os_utils import ensure_folder
from core.file_manager.savings import save_json


def save_params(params, save_dir):
    assert save_dir is not None

    ensure_folder(save_dir)

    filepaths = {'data': f'{save_dir}data_params.json',
                 'network': f'{save_dir}network_params.json',
                 'training': f'{save_dir}training_params.json'}

    for k in filepaths:
        save_json(params[k], filepaths[k])

    return filepaths
