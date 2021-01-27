from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical

from constants.config import HOMEMADE, MAX_WORD_SENTENCE

import numpy as np


def split_data(x_data, y_data, params):
    split_size = params['split_size']
    shuffle = params['shuffle']
    seed = params['seed']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        train_size=split_size,
                                                        shuffle=shuffle,
                                                        random_state=seed)

    return x_train, x_test, y_train, y_test


def create_dataset(sentence_series, target_series, tokenizer, tokenizer_type=HOMEMADE):
    assert len(sentence_series) == len(
        target_series), 'Error - create_dataset - sentence and target series have different length'

    length_samples = len(sentence_series)

    x_sentence, y_sentence = [], []

    if tokenizer_type == HOMEMADE:

        for i in range(length_samples):
            sentence = sentence_series.iloc[i]
            targets = target_series.iloc[i]

            x_sentence.append([tokenizer['word2idx'].get(word) for word in sentence.split(sep=' ')])
            y_sentence.append([tokenizer['label2idx'].get(t) for t in str(targets).split(sep=' ')])

    x_dataset = pad_sequences(sequences=x_sentence, maxlen=MAX_WORD_SENTENCE,
                              padding='post', value=tokenizer['word2idx']['PAD'])

    # y_dataset = pad_sequences(sequences=y_sentence, maxlen=MAX_WORD_SENTENCE,
    #                           padding='post', value=tokenizer['label2idx']['PAD'])

    num_classes = len(tokenizer['label2idx'])
    y_dataset = []
    for targets in y_sentence:
        cat_target = np.zeros(num_classes)
        for target in targets:
            cat_target += to_categorical(target, num_classes=num_classes)

        y_dataset.append(cat_target)

    y_dataset = np.array(y_dataset)

    return x_dataset, y_dataset
