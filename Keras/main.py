import Keras.preprocess as pre
import Keras.test as tes
import Keras.train as tra


def main(path_to_data):
    data = pre.load_data(path_to_data)
    data = pre.retype_data(data)
    X_train, X_test, Y_train, Y_test = pre.split_train_test(data)
    estimator = tra.create_estimator_NN2()
    # tra.kfold_model(estimator,
    #                         X=data.iloc[:, :-1],
    #                         Y=data.iloc[:, -1])
    fitted_model = tra.model_fit(estimator,
                                 X=X_train,
                                 Y=Y_train)
    predictions = tes.model_predict(fitted_model, X_test, Y_train)
    cm = tes.confusion_matrix(predictions, Y_test)
    print(cm)



if __name__ == '__main__':
    main(path_to_data="../_data/bank-additional-full.csv")
