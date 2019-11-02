from sklearn.metrics import confusion_matrix


def model_predict(fitted_model, X_test, Y_train):
    predictions_proba = fitted_model.predict(X_test)
    threshold = Y_train.sum()/Y_train.shape[0]
    predictions = [1 if prob >= threshold else 0 for prob in predictions_proba]
    return predictions


def confusion(target, prediction):
    cm = confusion_matrix(target, prediction)
    return cm

