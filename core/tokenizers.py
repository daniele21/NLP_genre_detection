def homemade_tokenizer(sentence_series, target_series):

    all_words = [word for sentence in sentence_series for word in sentence.split(sep=' ')]
    unique_words = set(all_words)

    all_targets = [target for targets in target_series for target in targets.split(sep=' ')]
    unique_targets = set(all_targets)

    # WORD2IDX
    word2idx = {w: i+2 for i, w in enumerate(unique_words)}
    word2idx['UNK'] = 1
    word2idx['PAD'] = 0

    # LABEL2IDX
    label2idx = {t: i for i, t in enumerate(unique_targets)}
    # label2idx['PAD'] = 0

    # IDX2WORD
    idx2word = {i: w for w, i in word2idx.items()}

    # IDX2LABEL
    idx2label = {i: t for t, i in label2idx.items()}

    return {'word2idx': word2idx,
            'idx2word': idx2word,
            'label2idx': label2idx,
            'idx2label': idx2label}
