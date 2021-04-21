from typing import Dict

from keras_preprocessing.sequence import pad_sequences
from pandas import Series
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical

from constants.config import MAX_WORD_SENTENCE

import numpy as np

from core.preprocessing.tokenizers import MyTokenizer


def split_data(x_data, y_data, params: Dict):
    split_size = params['split_size']
    shuffle = params['shuffle']
    seed = params['seed']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        train_size=split_size,
                                                        shuffle=shuffle,
                                                        random_state=seed)

    return x_train, x_test, y_train, y_test


def create_dataset(sentences: Series,
                   targets: Series,
                   tokenizer: MyTokenizer):

    assert len(sentences) == len(targets), 'Error - create_dataset - sentence and target series have different length'

    length_samples = len(sentences)

    x_sentence, y_sentence = [], []

    for i in range(length_samples):
        sentence = sentences.iloc[i]
        target = targets.iloc[i]

        x_sentence.append([tokenizer.word_to_index(word) for word in sentence.split(sep=' ') if word != ''])
        y_sentence.append([tokenizer.label_to_index(t) for t in str(target).split(sep=' ')])

    x_dataset = pad_sequences(sequences=x_sentence, maxlen=MAX_WORD_SENTENCE,
                              padding='post', value=tokenizer.word_to_index('PAD'))

    num_classes = tokenizer.n_labels
    y_dataset = []
    for targets in y_sentence:
        cat_target = np.zeros(num_classes)
        for target in targets:
            cat_target += to_categorical(target, num_classes=num_classes)

        y_dataset.append(cat_target)

    y_dataset = np.array(y_dataset)

    return x_dataset, y_dataset


def create_inference_dataset(sentences: Series,
                             tokenizer: MyTokenizer):

    sentences_list = []
    length_samples = len(sentences)

    for i in range(length_samples):
        sentence = sentences.iloc[i]
        tokenized_sentence = []
        for word in sentence.split(sep=' '):
            token = tokenizer.word_to_index(word)
            token = tokenizer.word_to_index('UNK') if token is None else token
            tokenized_sentence.append(token)
        sentences_list.append(tokenized_sentence)

    dataset = pad_sequences(sequences=sentences, maxlen=MAX_WORD_SENTENCE,
                            padding='post', value=tokenizer.word_to_index('PAD'))

    return dataset

