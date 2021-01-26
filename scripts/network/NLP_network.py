import keras

from scripts.constants.config import MAX_WORD_SENTENCE


def LSTM_network(params, compile=True):

    n_word_tokens = params['n_word_tokens']
    word_emb_size = params['word_emb_size']
    weight = params.get('weights')
    trainable = params['trainable']
    lstm_units = params['lstm_units']
    num_classes = params['n_classes']
    dropout_rate = params['dropout_rate']
    opt = params['optimizer']
    lr = params['lr']
    loss = params['loss']


    # INPUT
    sentence_input = keras.layers.Input(shape=(MAX_WORD_SENTENCE, ))

    # EMBEDDINGS
    embeddings = keras.layers.Embedding(input_dim=n_word_tokens,
                                        output_dim=word_emb_size,
                                        # weights=weight,
                                        # trainable=trainable
                                        )

    # LSTM
    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=lstm_units,
                                                              return_sequences=False))

    # DROPOUT
    dropout = keras.layers.Dropout(dropout_rate)
    # dropout = keras.layers.SpatialDropout1D(dropout_rate)

    # DENSE
    dense_layer = keras.layers.Dense(units=num_classes,
                                     activation='softmax')


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
