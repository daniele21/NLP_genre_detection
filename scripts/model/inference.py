import numpy as np
from copy import deepcopy

def model_inference(model, x, thr=0.5):
    y_pred = model.predict(x)

    y_bin_pred = deepcopy(y_pred)
    y_bin_pred[y_bin_pred >= thr] = 1
    y_bin_pred[y_bin_pred < thr] = 0

    return y_pred, y_bin_pred