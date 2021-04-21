from typing import Dict

from core.preprocessing.embeddings import load_pretrained_glove_embeddings
from core.preprocessing.tokenizers import MyTokenizer
from scripts.network.NLP_network import LSTM_network_pretrained_emb, LSTM_network


def init_network(params: Dict,
                 tokenizer: MyTokenizer,
                 compile: bool):

    if params['pretrained']:
        emb_path = params['emb_path']
        emb_weights = load_pretrained_glove_embeddings(tokenizer, embedding_path=emb_path)

        params['weights'] = emb_weights
        params['word_emb_size'] = 300
        nn = LSTM_network_pretrained_emb(params, compile=compile)

    else:
        nn = LSTM_network(params, compile=compile)

    return nn
