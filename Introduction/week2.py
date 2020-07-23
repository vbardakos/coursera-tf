"""
Contents:
    input_fn -> fashion mnist
    LossCallback -> loss < .3
    DenseNet -> Dense Network
"""
from tensorflow import keras
import numpy as np
import tensorflow as tf


def input_fn():
    mnist = keras.datasets.fashion_mnist
    (x0, y0), (x1, y1) = mnist.load_data()
    x0 = x0/255.
    x1 = x1/255.
    return (x0, y0), (x1, y1)


class LossCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.3:
            print("\nLossCallback : Reached 70% accuracy"*3)
            self.model.stop_training = True


class DenseNet(object):

    def __init__(self, features, labels):
        self.x = features
        self.y = labels
        self.model = None

    def fit(self):
        # build our model
        mdl_input = keras.Input(self.x.shape[1:])
        mdl_layer = keras.layers.Flatten()(mdl_input)
        mdl_layer = keras.layers.Dense(128, activation='relu')(mdl_layer)
        mdl_logit = keras.layers.Dense(10, activation='softmax')(mdl_layer)
        # compile and fit
        model = keras.Model(mdl_input, mdl_logit)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        my_callback = LossCallback()  # callback @0.3
        model.fit(self.x, self.y, epochs=6, callbacks=[my_callback])
        self.model = model
        return self

    def eval(self, features, labels):
        if self.model is None:
            self.fit()
        evaluation = self.model.evaluate(features, labels)
        print("Evaluation Score is : {}".format(evaluation))

    def predict(self, features, labels):
        if self.model is None:
            self.fit()
        predictions = self.model.predict(features)
        predictions = np.argmax(predictions, axis=1)
        mtx = tf.math.confusion_matrix(labels, predictions)
        print("Confusion Matrix : ", mtx, sep="\n")


# get data
(x_train, y_train), (x_test, y_test) = input_fn()

if __name__ == '__main__':
    DenseNet(x_train, y_train).eval(x_test, y_test)