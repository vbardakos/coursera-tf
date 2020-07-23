"""
Contents:
    input_fn -> fashion mnist
    AccCallback -> min loss or max acc
    ConvNet -> Convolution Network
"""

from tensorflow import keras
import numpy as np


def input_fn():
    data = keras.datasets.fashion_mnist
    (x0, y0), (x1, y1) = data.load_data()

    def mapper(x):
        return np.expand_dims(x / 255., axis=-1)

    x0, x1 = mapper(x0), mapper(x1)
    return (x0, y0), (x1, y1)


class AccCallBack(keras.callbacks.Callback):

    def __init__(self, patience=10, acc_rate=.9):
        super(AccCallBack, self).__init__()
        self.patience = patience
        self.acc = acc_rate
        self.wait = 0  # compared to patience
        self.best_loss = np.inf  # holds best recorded loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if logs.get('accuracy') < self.acc and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        elif current_loss > self.best_loss:
            self.wait += 1
            if self.wait > self.patience:
                print(f"\nLossCallback : Loss started increasing at {self.wait} with patience {self.patience}" * 3)
                self.model.stop_training = True
        else:
            print(f"\nAccuracyCallback : Model reached {self.acc * 100}% accuracy" * 3)
            self.model.stop_training = True


class ConvNet(object):

    def __init__(self, features, labels, callback=AccCallBack()):
        self.x = features
        self.y = labels
        self.callback = callback

    def fit(self):
        # build model
        mdl_input = keras.layers.Input(self.x.shape[1:])
        mdl_layer = keras.layers.Conv2D(64, (3, 3))(mdl_input)
        mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
        mdl_layer = keras.layers.Conv2D(64, (3, 3))(mdl_layer)
        mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
        mdl_layer = keras.layers.Flatten()(mdl_layer)
        mdl_layer = keras.layers.Dense(128, activation='relu')(mdl_layer)
        mdl_logit = keras.layers.Dense(10, activation='softmax')(mdl_layer)
        # compile and fit
        model = keras.Model(mdl_input, mdl_logit)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.x, self.y, epochs=10, callbacks=[self.callback], verbose=0)
        return model

    def eval(self, features, labels):
        model = self.fit()
        evaluation = model.evaluate(features, labels)
        print(f"Evaluation reached {evaluation}")


if __name__ == '__main__':
    # get data
    (x_train, y_train), (x_test, y_test) = input_fn()
    # train and eval
    ConvNet(x_train, y_train).fit()
