import pandas as pd

from core.text_preprocessing import remove_punctuations, stem_text, lemmatize_text

def stem_sentence(sentence):
    new_sentence = ''
    for word in sentence.split(sep=' '):
        new_sentence += f'{stem_text(word)} '

    return new_sentence


def lemmatize_sentence(sentence):
    new_sentence = ''
    for word in sentence.split(sep=' '):
        new_sentence += f'{lemmatize_text(word)}'

    return new_sentence


def sentence_preprocessing(input_data, stemming=True,
                                       lemmatization=False,
                                       lowercase=True,
                                       preload=None):

    if(preload is not None):
        return pd.read_csv(preload, index_col=0)

    data = input_data.copy(deep=True)

    # REMOVING PUNCTUATIONS
    data['synopsis'] = data['synopsis'].apply(remove_punctuations)

    # STEMMING
    if(stemming):
        data['synopsis'] = data['synopsis'].apply(stem_sentence)

    # LEMMATIZATION
    if(lemmatization):
        data['synopsis'] = data['synopsis'].apply(lemmatize_sentence)

    # LOWERING
    if(lowercase):
        data['synopsis'] = data['synopsis'].apply(lambda x: str(x).lower())

    return data


