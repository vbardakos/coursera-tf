import os
import requests
from PIL import UnidentifiedImageError

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import RMSprop


def downloader(url: str, path: str, ignore=False):
    if not path:
        path = os.getcwd()

    if os.path.isdir(path):
        file = url.split('/')[-1]
        path = os.path.normpath(os.path.join(path, file))

    if os.path.exists(path) and not ignore:
        print("Warning: File already exists -- Try `ignore` param")
    else:
        r = requests.get(url, verify=False)
        with open(path, 'wb') as f:
            f.write(r.content)
    return path


def corrupt_checker(path: str):
    if os.path.isdir(path):
        del_files = list()
        for f in os.listdir(path):
            file = os.path.join(path, f)
            if os.path.getsize(file):
                try:
                    keras.preprocessing.image.load_img(file)
                except UnidentifiedImageError:
                    os.remove(file)
                    del_files.append(file)
                    print(f"{file} got deleted")
            else:
                os.remove(file)
                del_files.append(file)
                print(f"{file} got deleted")
        print(f"from {len(os.listdir(path))} in total, {len(del_files)} deleted")
    else:
        raise Exception("Path is not a directory")
    return del_files


def input_gen(path: str, size: (tuple, list)):
    # train generator
    generator = ImageDataGenerator(rescale=1 / 255.,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   zoom_range=.2,
                                   shear_range=.2,
                                   height_shift_range=.2,
                                   width_shift_range=.2,
                                   validation_split=.2
                                   )
    train_gen = generator.flow_from_directory(path,
                                              target_size=size,
                                              batch_size=32,
                                              class_mode='binary',
                                              subset='training')
    # validation generator
    valid_gen = generator.flow_from_directory(path,
                                              target_size=size,
                                              batch_size=32,
                                              class_mode='binary',
                                              subset='validation')
    return train_gen, valid_gen


def pretrained_inception(weights_path):
    pretrained = InceptionV3(input_shape=(100, 100, 3),
                             include_top=False,
                             weights=None)
    # load weights
    pretrained.load_weights(weights_path)
    # set all layers to non-trainable
    for layer in pretrained.layers:
        layer.trainable = False
    # get layer to build on - str from summary 'layer type'
    n_input = pretrained.get_layer('mixed7')
    # build extra layers
    n_layer = keras.layers.Flatten()(n_input.output)
    n_layer = keras.layers.Dense(1024, activation='relu')(n_layer)
    n_logit = keras.layers.Dense(1, activation='sigmoid')(n_layer)
    # build & compile
    model = keras.Model(pretrained.input, n_logit)
    model.compile(optimizer=RMSprop(lr=.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # # download weights
    my_link = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    my_path = ''
    f_path = downloader(my_link, my_path)

    # import data generators
    corrupt_checker('tmp/dogs-vs-cats/train/dog/')
    corrupt_checker('tmp/dogs-vs-cats/train/cat/')
    data_path = 'tmp/dogs-vs-cats/train/'
    train, validation = input_gen(data_path, size=(100, 100))

    # model
    model = pretrained_inception(f_path)
    model.fit_generator(train,
                        validation_data=validation,
                        steps_per_epoch=20000 // 32,
                        epochs=2,
                        verbose=1)
