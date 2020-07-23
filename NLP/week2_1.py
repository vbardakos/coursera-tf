import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def input_fn(ds_name):
    ds, info = tfds.load(ds_name, with_info=True, as_supervised=True)
    train_ds, valid_ds = ds['train'], ds['test']
    return train_ds, valid_ds, info


def preprocess(ds):
    txt_num, lbl_num = list(), list()

    for t0, l0 in ds:
        txt_num.append(t0.numpy().decode('utf8'))
        lbl_num.append(l0.numpy())

    lbl_num = np.array(lbl_num)

    # txt_num = ds.map(lambda x, y: x)
    # lbl_num = ds.map(lambda x, y: y)
    # txt_num = map(lambda x: x.numpy().decode('utf8'), txt_num)
    # lbl_num = np.fromiter(lbl_num, 'int')
    return txt_num, lbl_num


VOC = 10000
OOV = '<OOV>'
DIM = 16
MAX = 120
TRC = 'post'


class NlpModel(object):

    def __init__(self, train, valid):
        self.txt_train = train[0]
        self.lbl_train = train[1]
        self.txt_valid = valid[0]
        self.lbl_valid = valid[1]
        self.w_idx = None

    def tokenize(self):

        tkn = Tokenizer(num_words=VOC, oov_token=OOV)
        tkn.fit_on_texts(self.txt_train)
        self.w_idx = tkn.word_index

        seq_train = tkn.texts_to_sequences(self.txt_train)
        pad_train = pad_sequences(seq_train, maxlen=MAX, truncating=TRC)

        seq_valid = tkn.texts_to_sequences(self.txt_valid)
        pad_valid = pad_sequences(seq_valid, maxlen=MAX, truncating=TRC)

        return pad_train, pad_valid

    def fit_eval(self, epochs: int):
        txt_train, txt_valid = self.tokenize()
        n_model = self.architecture()

        history = n_model.fit(txt_train,
                              self.lbl_train,
                              epochs=epochs,
                              validation_data=(txt_valid, self.lbl_valid))
        return history

    @property
    def word_index(self):
        return self.w_idx

    @property
    def summary(self):
        return self.architecture().summary()

    @staticmethod
    def architecture():
        n_input = keras.Input(MAX)
        n_layer = keras.layers.Embedding(VOC, DIM)(n_input)
        n_layer = keras.layers.GlobalAvgPool1D()(n_layer)
        n_layer = keras.layers.Dense(6, activation='relu')(n_layer)
        n_logit = keras.layers.Dense(1, activation='sigmoid')(n_layer)

        n_model = keras.Model(n_input, n_logit)
        n_model.compile(optimizer=RMSprop(lr=.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return n_model


if __name__ == '__main__':
    # get data & preprocess
    ds_train, ds_valid, meta = input_fn('imdb_reviews')
    train_set = preprocess(ds_train)
    valid_set = preprocess(ds_valid)

    # model fit
    my_model = NlpModel(train=train_set, valid=valid_set)
    my_history = my_model.fit_eval(10)