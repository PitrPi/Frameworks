# or tensorflow.keras
import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(16, input_dim=20, activation='relu')
        self.d2 = Dense(8, activation='relu')
        self.d3 = Dense(1, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


def create_model():
    # Create an instance of the model
    model = MyModel()
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    return model, loss_object, optimizer


# @tf.function
def train_step(data, labels, model, loss_object, optimizer):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


# @tf.function
def test_step(data, labels, model, loss_object):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    predictions = model(data)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


def train_model(data, labels, test_data, test_labels,
                train_loss, train_accuracy, test_loss, test_accuracy):
    EPOCHS = 5
    for epoch in range(EPOCHS):
        for data, labels in train_ds:
            train_step(data, labels)

        for test_data, test_labels in test_ds:
            test_step(test_data, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()