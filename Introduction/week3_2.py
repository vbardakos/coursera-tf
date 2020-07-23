"""
Contents : conv_inspect -> shows outputs by conv layer // IdxError
"""
from .week3_1 import *
import matplotlib.pyplot as plt


def conv_inspect(conv_num, test: tuple,
                 keras_model, label: int = None):
    """
    IndexError: index 1 is out of bounds for axis 3 with size 1
    """
    x, y = test
    f, arr = plt.subplots(3, 4)
    if label in y:
        label = np.where(y == label)
        n0, n1, n2 = label[0], label[1], label[2]
    else:
        n0, n1, n2 = 0, 1, 2

    print(n0, n1, n2, y, sep="\n")

    layer_outputs = [layer.output for layer in keras_model.layers]
    activation_model = keras.models.Model(keras_model.input, layer_outputs)

    for img in range(0, 4):
        f1 = activation_model.predict(x[n0].reshape(1, 28, 28, 1))[img]
        arr[0, img].imshow(f1[0, :, :, conv_num], cmap='inferno')
        arr[0, img].grid(False)
        f2 = activation_model.predict(x[n1].reshape(1, 28, 28, 1))[img]
        arr[1, img].imshow(f2[0, :, :, conv_num], cmap='inferno')
        arr[1, img].grid(False)
        f3 = activation_model.predict(x[n2].reshape(1, 28, 28, 1))[img]
        arr[2, img].imshow(f3[0, :, :, conv_num], cmap='inferno')
        arr[2, img].grid(False)
    plt.show()


if __name__ == 'main':
    (x_train, y_train), (x_test, y_test) = input_fn()
    model = ConvNet(x_train, y_train).fit()
    # conv_inspect try
