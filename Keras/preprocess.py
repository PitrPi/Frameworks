import pandas as pd
from sklearn.model_selection import train_test_split



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
    return train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                            test_size=0.7, random_state=42)


if __name__ == '__main__':
    data = load_data("_data/bank-additional-full.csv")
    train, test = split_train_test(data)
