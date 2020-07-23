import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from Introduction.week3_2 import AccCallBack


def error_handler(path):
    """
    Checks if there are corrupted files in our data

    :param path: data path
    """
    from tqdm import tqdm
    from PIL import Image
    os.chdir(path)
    for file in tqdm(os.listdir()):
        # noinspection PyBroadException
        try:
            img = Image.open(file)
            np.asarray(img)
        except:
            os.remove(file)
            print(f"Warning : {file} removed")


class GenConvNet(object):

    def __init__(self, path: str):
        self.mdl = None
        self.path = path
        self.call = None
        self.train = None
        self.valid = None

    def data_to_gen(self, batch: int, size=(150, 150), mode='binary'):
        generator = ImageDataGenerator(rescale=1 / 255., validation_split=.2)
        self.train = generator.flow_from_directory(self.path,
                                                   target_size=size,
                                                   batch_size=batch,
                                                   class_mode=mode)
        self.valid = generator.flow_from_directory(self.path,
                                                   target_size=size,
                                                   batch_size=batch,
                                                   class_mode=mode)

        return self

    def architecture(self, get_weights=False, weights_path="GenConvNet.hdf5"):
        # checkpoint callback
        self.call = keras.callbacks.ModelCheckpoint(weights_path,
                                                    save_best_only=True)
        # model architecture
        mdl_input = keras.Input(self.train.image_shape)
        mdl_layer = keras.layers.Conv2D(16, (3, 3), activation='relu')(mdl_input)
        mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
        mdl_layer = keras.layers.Conv2D(32, (3, 3), activation='relu')(mdl_layer)
        mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
        mdl_layer = keras.layers.Conv2D(64, (3, 3), activation='relu')(mdl_layer)
        mdl_layer = keras.layers.MaxPool2D(2, 2)(mdl_layer)
        mdl_layer = keras.layers.Flatten()(mdl_layer)
        mdl_layer = keras.layers.Dense(512, activation='relu')(mdl_layer)
        mdl_layer = keras.layers.Dense(64, activation='relu')(mdl_layer)
        # if we didn't know how many labels we have we could use
        # max(self.train.class_indices.values()) - it starts from 0
        mdl_logit = keras.layers.Dense(1, activation='sigmoid')(mdl_layer)
        # model build and compile
        model = keras.Model(mdl_input, mdl_logit)
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # load weights
        if get_weights:
            try:
                model.load_weights(weights_path)
            except FileNotFoundError:
                pass

        self.mdl = model
        return self

    def fit(self, epochs=5):
        # accuracy / loss callback
        acc_callback = AccCallBack(patience=1, acc_rate=.95)

        if not self.mdl:
            self.architecture()

        # specify allocation strategy
        strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

        with strategy.scope():
            model = self.mdl

        # model fit
        mdl_history = model.fit(self.train,
                                steps_per_epoch=500,
                                epochs=epochs,
                                validation_data=self.valid,
                                callbacks=[self.call, acc_callback],
                                verbose=2)
        return mdl_history

    def predict(self, path, batch=50):
        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            files = path.split('/')[-1]

        self.architecture(get_weights=True)
        for file in files:
            image = keras.preprocessing.image.load_img(f"{path}/{file}", target_size=(150, 150))
            image = keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = np.vstack([image])

            label = self.mdl.predict(image, batch_size=batch)

            if label[0] == 0:
                print(f"{file} is a cat")
            else:
                print(f"{file} is a dog")


if __name__ == '__main__':
    # load data as generator
    train_path = 'tmp/dogs-vs-cats/train'

    # ConvNet
    net = GenConvNet(path=train_path)
    history = net.data_to_gen(batch=50).fit(epochs=10)
    # predict
    # test_path = 'tmp/dogs-vs-cats/test1/test1'
    # net.data_to_gen(batch=1).predict('samples')
