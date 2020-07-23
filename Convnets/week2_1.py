import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def input_gen_fn(train_path: str, valid_path: str = None):
    if valid_path:
        paths = [train_path, valid_path]
    else:
        paths = [train_path]

    gen_list = np.zeros(2, dtype='object')
    for idx, path in enumerate(paths):
        gen_list[idx] = ImageDataGenerator(rescale=1 / 255.,
                                           horizontal_flip=True,
                                           shear_range=.2,
                                           rotation_range=.45,
                                           width_shift_range=.2,
                                           height_shift_range=.2,
                                           zoom_range=.2,
                                           fill_mode='nearest')
        gen_list[idx] = gen_list[idx].flow_from_directory(path,
                                                          target_size=(300, 300),
                                                          batch_size=128,
                                                          class_mode='binary')
    return gen_list


def gen_convnet(shape: (list, tuple) = (300, 300, 3)):
    # model architecture
    mdl_input = keras.Input(shape)
    mdl_layer = keras.layers.Conv2D(16, (3, 3), activation='relu')(mdl_input)
    mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
    mdl_layer = keras.layers.Conv2D(32, (3, 3), activation='relu')(mdl_layer)
    mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
    mdl_layer = keras.layers.Conv2D(64, (3, 3), activation='relu')(mdl_layer)
    mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
    mdl_layer = keras.layers.Flatten()(mdl_layer)
    mdl_layer = keras.layers.Dense(512, activation='relu')(mdl_layer)
    mdl_layer = keras.layers.Dense(64, activation='relu')(mdl_layer)
    # max(self.train.class_indices.values()) - it starts from 0
    mdl_logit = keras.layers.Dense(1, activation='sigmoid')(mdl_layer)

    # model build and compile
    model = keras.Model(mdl_input, mdl_logit)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def acc_loss_plot(mdl_history):
    """
    Plots model accuracy, validation accuracy, loss and validation loss in respect with epochs

    :param mdl_history: trained model
    """
    acc = mdl_history.history['accuracy']
    loss = mdl_history.history['loss']
    try:
        val_acc = mdl_history.history['val_accuracy']
        val_loss = mdl_history.history['val_loss']
        validation = True
    except KeyError:
        validation = False

    epochs = range(len(acc))
    # plot accuracy figure
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    if validation:
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
    else:
        plt.title('Training accuracy')
    plt.legend()
    plt.figure()
    # plot loss figure
    plt.plot(epochs, loss, 'r', label='Training Loss')
    if validation:
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
    else:
        plt.title('Training loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # load data from paths
    t_path = '../Introduction/tmp/horse-or-human/'
    v_path = '../Introduction/tmp/validation-horse-or-human/'
    gen_train, gen_valid = input_gen_fn(t_path, v_path)

    # Convnet
    model = gen_convnet()
    history = model.fit(gen_train,
                        steps_per_epoch=8,
                        epochs=1,
                        # validation_data=gen_valid,
                        verbose=2)
    # visualise accuracy and loss
    acc_loss_plot(history)
