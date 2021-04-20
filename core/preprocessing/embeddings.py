from tqdm import tqdm

from constants.config import EMBEDDING_DIM
from constants.paths import GLOVE_PATH
import numpy as np

def load_pretrained_glove_embeddings(tokenizer):
    embeddings_index = {}

    f = open(GLOVE_PATH)

    for line in tqdm(f, desc='> Loading Embeddings'):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(tokenizer['word2idx']), EMBEDDING_DIM))
    for word, i in tokenizer['word2idx'].items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
