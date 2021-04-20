from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam

from scripts.network.NLP_network import LSTM_network


def train_model(data_resources, params, callbacks=None):
    network_params = params['network']
    training_params = params['training']

    tokenizer = data_resources['tokenizer']
    dataset = data_resources['dataset']

    network_params['n_word_tokens'] = len(tokenizer['word2idx'])
    network_params['n_classes'] = len(tokenizer['label2idx'])
    network_params['optimizer'] = Adam
    network_params['loss'] = BinaryCrossentropy(from_logits=True)
    # network_params['n_classes'] = 40
    model = LSTM_network(network_params, compile=True)


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


