import numpy as np
from copy import deepcopy

def model_inference(model, x, thr=0.5):
    y_pred = model.predict(x)

    y_bin_pred = deepcopy(y_pred)
    y_bin_pred[y_bin_pred >= thr] = 1
    y_bin_pred[y_bin_pred < thr] = 0

    return y_pred, y_bin_pred

def extract_five_movies(y_pred, tokenizer):

    five_movies_pred = y_pred.argsort()
    five_movies_pred = np.flip(five_movies_pred[:,-5:])

    labels = []
    for y in five_movies_pred:
        labels.append([tokenizer['idx2label'][str(token)] for token in y])

    final_labels = []
    for row in labels:
        row_value = ''
        for label in row:
            row_value += f'{label} '
        row_value = row_value.rstrip()

        final_labels.append(row_value)

    return final_labels