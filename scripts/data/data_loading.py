import pandas as pd


def load_data(data_path):

    data = pd.read_csv(data_path, index_col=0)

    return data

