from core.text_preprocessing import init_nltk
from core.tokenizers import homemade_tokenizer
from scripts.constants.config import HOMEMADE
from scripts.data.data_loading import load_data
from scripts.data.dataset import split_data, create_dataset
from scripts.data.preprocessing import sentence_preprocessing


def generate_training_dataset(params):
    """

    :param params:  dict {
                          'train':      bool,
                          'split_size:  double,
                          'shuffle':    bool,
                          'seed':       int
                          }
    :return:
        x_train, x_test, y_train, y_test
    """
    data = load_data(params)

    init_nltk()
    prep_data = sentence_preprocessing(data,
                                       stemming=True,
                                       lemmatization=False,
                                       lowercase=True,
                                       preload=params.get('preload'))

    tokenizer = homemade_tokenizer(prep_data['synopsis'], prep_data['genres'])

    x_dataset, y_dataset = create_dataset(prep_data['synopsis'], prep_data['genres'], tokenizer, HOMEMADE)

    x_train, x_test, y_train, y_test = split_data(x_dataset, y_dataset, params)

    print(f'X_Train: {x_train.shape}\t-\t X_Test : {x_test.shape}')
    print(f'y_Train: {y_train.shape}\t-\t y_Test : {y_test.shape}')

    dataset = {'train':{'x': x_train,
                        'y': y_train},
               'test': {'x': x_test,
                        'y': y_test}}

    return {'tokenizer': tokenizer,
            'dataset': dataset}
