# TODO : Check the original to verify

from tensorflow import keras


def mobile_net():
    n_inputs = keras.Input((300, 300, 3))

    n_layer0 = keras.layers.Conv2D(8, (3, 3))(n_inputs)
    n_layer0 = keras.layers.BatchNormalization()(n_layer0)
    n_layer0 = keras.layers.ReLU()(n_layer0)
    n_layer0 = keras.layers.DepthwiseConv2D((3, 3))(n_layer0)
    n_layer0 = keras.layers.BatchNormalization()(n_layer0)
    n_layer0 = keras.layers.ReLU()(n_layer0)
    n_layer0 = keras.layers.Conv2D(1, (1, 1))(n_layer0)
    n_layer0 = keras.layers.BatchNormalization()(n_layer0)
    n_layer0 = keras.layers.ReLU()(n_layer0)

    n_layer1 = keras.layers.Flatten()(n_layer0)
    n_layer1 = keras.layers.Dense(1, activation='sigmoid')(n_layer1)

    model = keras.Model(n_inputs, n_layer1)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def input_gen():
    train_path = '../Introduction/tmp/horse-or-human'
    train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,
                                                             horizontal_flip=True,
                                                             zoom_range=.2,
                                                             rotation_range=.2,
                                                             shear_range=.2,
                                                             fill_mode='nearest')
    train_gen = train_gen.flow_from_directory(train_path,
                                              target_size=(300, 300),
                                              batch_size=20,
                                              class_mode='binary')
    valid_path = '../Introduction/tmp/validation-horse-or-human'
    valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
    valid_gen = valid_gen.flow_from_directory(valid_path,
                                              batch_size=20,
                                              target_size=(300, 300),
                                              class_mode='binary')

    return train_gen, valid_gen


if __name__ == '__main__':
    train, valid = input_gen()

    my_model = mobile_net()
    print(my_model.layers)
    # my_model.fit(train,
    #              steps_per_epoch=50,
    #              epochs=5,
    #              validation_data=valid)