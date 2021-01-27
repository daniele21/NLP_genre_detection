import pandas as pd

from constants.paths import TRAIN_PATH, TEST_PATH


def load_data(params):

    train = params['train']

    if(train):
        data = pd.read_csv(TRAIN_PATH, index_col=0)


    else:
        data = pd.read_csv(TEST_PATH, index_col=0)



    return data

