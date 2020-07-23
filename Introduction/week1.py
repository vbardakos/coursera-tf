"""
Contents : Single Dense Network
"""
import numpy as np
from tensorflow import keras


class SingleDense(object):
    """
    Week 1: Hello World Example Model
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def fit(self):
        # build the model
        mdl_input = keras.Input(shape=[1])
        mdl_logit = keras.layers.Dense(1)(mdl_input)
        model = keras.Model(mdl_input, mdl_logit)
        # compile & fit model
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(self.x, self.y, epochs=500)
        return model

    def predict(self, num):
        model = self.fit()
        prediction = model.predict([num])
        print(f"True value of : {num * 2 - 1}")
        print(f"My prediction of {num} : {int(np.ceil(prediction))}")


# SingleDense data
xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

if __name__ == '__main__':
    SingleDense(xs, ys).predict(10)