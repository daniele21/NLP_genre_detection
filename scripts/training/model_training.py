from typing import Dict, Text

from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from core.preprocessing.tokenizers import MyTokenizer
from scripts.network.NLP_network import LSTM_network
from scripts.network.network_init import init_network


def train_model(dataset: Dict[Text, Text],
                tokenizer: MyTokenizer,
                params: Dict,
                callbacks=None):

    network_params = params['network']
    training_params = params['training']

    network_params['n_word_tokens'] = tokenizer.n_words + 1
    network_params['n_classes'] = tokenizer.n_labels

    network_params['optimizer'] = Adam
    network_params['loss'] = BinaryCrossentropy(from_logits=True)

    model = init_network(network_params, tokenizer, compile=True)
    # model = LSTM_network(network_params, compile=True)

    x_train, y_train = dataset['train']['x'], dataset['train']['y']
    x_test, y_test = dataset['test']['x'], dataset['test']['y']

    batch_size = training_params['batch_size']
    epochs = training_params['epochs']

    model.fit(x=x_train, y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=callbacks,
              verbose=1,
              )

    return model


