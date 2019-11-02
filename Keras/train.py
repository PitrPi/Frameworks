from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# define base model
def NN2_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=20,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, input_dim=16,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def create_estimator_NN2():
    estimator = KerasRegressor(build_fn=NN2_model,
                               epochs=5,
                               batch_size=100,
                               verbose=1)
    return estimator


def kfold_model(estimator, X, Y):
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def model_fit(model, X, Y):
    model.fit(X, Y, epochs=50)
    return model


