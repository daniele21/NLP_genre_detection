from typing import Text, Dict

from core.file_manager.savings import save_json, pickle_save
from core.preprocessing.text_preprocessing import init_nltk
from core.preprocessing.tokenizers import init_tokenizer
from constants.config import HOMEMADE
from scripts.data.data_loading import load_data
from scripts.data.dataset import split_data, create_dataset, create_inference_dataset
from scripts.data.preprocessing import sentence_preprocessing

import numpy as np


def generate_training_dataset(params: Dict,
                              save_dir: Text =None):
    """

    :param save_dir:
    :param params:  dict {
                          'train':      True,
                          'split_size:  double,
                          'shuffle':    bool,
                          'seed':       int
                          }
    :return:
        x_train, x_test, y_train, y_test
    """
    data = load_data(params['data_path'])

    init_nltk()
    prep_data = sentence_preprocessing(data,
                                       stemming=True,
                                       lemmatization=False,
                                       lowercase=True,
                                       stopwords=True,
                                       preload=params.get('preload'),
                                       )

    tokenizer = init_tokenizer(params['tokenizer'])
    tokenizer.fit(prep_data['synopsis'], prep_data['genres'])

    x_dataset, y_dataset = create_dataset(prep_data['synopsis'], prep_data['genres'], tokenizer)

    x_train, x_test, y_train, y_test = split_data(x_dataset, y_dataset, params)

    print(f'X_Train: {x_train.shape}\t-\t X_Test : {x_test.shape}')
    print(f'y_Train: {y_train.shape}\t-\t y_Test : {y_test.shape}')

    dataset = {'train': {'x': x_train,
                         'y': y_train},
               'test': {'x': x_test,
                        'y': y_test}}

    if save_dir is not None:
        filepath = f'{save_dir}tokenizer'
        pickle_save(tokenizer, filepath)

    return dataset, tokenizer


def generate_test_dataset(data_path, tokenizer):

    data = load_data(data_path)
    init_nltk()
    prep_data = sentence_preprocessing(data,
                                       stemming=True,
                                       lemmatization=False,
                                       lowercase=True)

    dataset = create_inference_dataset(prep_data['synopsis'], tokenizer)

    return data, np.array(dataset)
