import TF2.preprocess as pre
import TF2.test as tes
import TF2.train as tra


def main(path_to_data):
    data = pre.load_data(path_to_data)
    data = pre.retype_data(data)
    X_train, X_test, Y_train, Y_test = pre.split_train_test(data)
    model, loss_object, optimizer = tra.create_model()
    tra.train_model(X_train.values, Y_train.values,
                    X_test.values, Y_train.values.model, loss_object, optimizer)

    fitted_model = tra.model_fit(estimator,
                                 X=X_train,
                                 Y=Y_train)
    predictions = tes.model_predict(fitted_model, X_test.values, Y_train.values)
    cm = tes.confusion_matrix(predictions, Y_test)
    print(cm)


if __name__ == '__main__':
    main(path_to_data="../_data/bank-additional-full.csv")