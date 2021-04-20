import keras

from constants.config import MAX_WORD_SENTENCE
from scripts.network.network_utils import get_optimizer, get_loss

def LSTM_network(params, compile=True):

    n_word_tokens = params['n_word_tokens']
    word_emb_size = params['word_emb_size']
    lstm_units = params['lstm_units']
    num_classes = params['n_classes']
    dropout_rate = params['dropout_rate']
    opt = get_optimizer(params['optimizer'])
    lr = params['lr']
    loss = get_loss(params['loss'])


    # INPUT
    sentence_input = keras.layers.Input(shape=(MAX_WORD_SENTENCE, ))

    # EMBEDDINGS
    embeddings = keras.layers.Embedding(input_dim=n_word_tokens,
                                        output_dim=word_emb_size
                                        )

    # LSTM
    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=lstm_units,
                                                              return_sequences=False))

    # DROPOUT
    dropout = keras.layers.Dropout(dropout_rate)
    # dropout = keras.layers.SpatialDropout1D(dropout_rate)

    # DENSE
    dense_layer = keras.layers.Dense(units=num_classes,
                                     activation='sigmoid')


    # FORWARDING
    x = embeddings(sentence_input)
    h = dropout(x)
    h = lstm_layer(h)
    out = dense_layer(h)

    model = keras.models.Model(inputs=sentence_input,
                               outputs=out)

    print(model.summary())

    if(compile):
        model.compile(optimizer=opt(learning_rate=lr),
                      loss=loss,
                      metrics=['acc'],
                      )

    return model

def LSTM_network_pretrained_emb(params, compile=True):

    n_word_tokens = params['n_word_tokens']
    word_emb_size = params['word_emb_size']
    weights = params['weights']
    lstm_units = params['lstm_units']
    num_classes = params['n_classes']
    dropout_rate = params['dropout_rate']
    opt = get_optimizer(params['optimizer'])
    lr = params['lr']
    loss = get_loss(params['loss'])


    # INPUT
    sentence_input = keras.layers.Input(shape=(MAX_WORD_SENTENCE, ))

    # EMBEDDINGS
    embeddings = keras.layers.Embedding(input_dim=n_word_tokens,
                                        output_dim=word_emb_size,
                                        weights=[weights],
                                        trainable=False
                                        )

    # LSTM
    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=lstm_units,
                                                              return_sequences=False))

    # DROPOUT
    dropout = keras.layers.Dropout(dropout_rate)
    # dropout = keras.layers.SpatialDropout1D(dropout_rate)

    # DENSE
    dense_layer = keras.layers.Dense(units=num_classes,
                                     activation='sigmoid')


    # FORWARDING
    x = embeddings(sentence_input)
    h = dropout(x)
    h = lstm_layer(h)
    out = dense_layer(h)

    model = keras.models.Model(inputs=sentence_input,
                               outputs=out)

    print(model.summary())

    if(compile):
        model.compile(optimizer=opt(learning_rate=lr),
                      loss=loss,
                      metrics=['acc'],
                      )

    return model
