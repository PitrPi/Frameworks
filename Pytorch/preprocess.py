import math

import pandas as pd
from torch.utils.data import random_split


def load_data(path):
    with open(path) as f:
        data = pd.read_csv(f, sep=";")
    return data


def retype_data(data):
    for pos, typ in enumerate(data.dtypes):
        if typ == 'object':
            data.iloc[:, pos] = pd.factorize(data.iloc[:, pos])[0]
    return data.apply(pd.to_numeric, downcast='float')


def split_train_test(data):
    return random_split(data, [math.floor(data.shape[0]*0.7), math.ceil(data.shape[0]*0.3)])


if __name__ == '__main__':
    data = load_data("_data/bank-additional-full.csv")
    train, test = split_train_test(data)
