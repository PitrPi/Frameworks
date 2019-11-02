import Pytorch.preprocess as pre
import Pytorch.test as tes
import Pytorch.train as tra


def main(path_to_data):
    data = pre.load_data(path_to_data)
    data = pre.retype_data(data)
    train_data, test_data = pre.split_train_test(data)
    model = tra.train_linear(train_data)
    input = tes.test_linear(model, test_data)
    input = out.output(input)


if __name__ == '__main__':
    main(path_to_data="../_data/bank-additional-full.csv")
