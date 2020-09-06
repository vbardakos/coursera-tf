import json
import numpy as np
from typing import Tuple
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def input_fn(json_path: str, train_split: float = 0.0):
    datastore, validation = np.empty(0), np.empty(0)

    with open(json_path, 'r') as f:
        for line in f:
            datastore = np.append(datastore, json.loads(line))

    features = np.vectorize(lambda x: x['headline'])(datastore)
    labels = np.vectorize(lambda x: x['is_sarcastic'])(datastore)

    if 0 < train_split < 1:
        features = np.split(features, [int(train_split * features.shape[0])])
        labels = np.split(labels, [int(train_split * labels.shape[0])])

    return features, labels


def preprocess(features: np.ndarray, word_num: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    if len(features) == 2:
        features, validation = features
    else:
        validation = 0

    tkn = Tokenizer(num_words=word_num, oov_token="<OOV>")
    tkn.fit_on_texts(features)
    features = tkn.texts_to_sequences(features)
    features = pad_sequences(features, maxlen=seq_length, padding='post', truncating='post')

    if isinstance(validation, np.ndarray):
        validation = tkn.texts_to_sequences(validation)
        validation = pad_sequences(validation, maxlen=seq_length, padding='post', truncating='post')

    return features, validation, tkn.word_index


def architecture(word_length: int, vocabulary: int, dimensions: int) -> keras.Model:
    inputs = keras.Input(shape=[word_length], dtype='int64')
    layers = keras.layers.Embedding(vocabulary, dimensions)(inputs)
    layers = keras.layers.Dense(128, activation='relu')(layers)
    layers = keras.layers.GlobalAveragePooling1D()(layers)
    layers = keras.layers.Dense(64, activation='relu')(layers)
    logits = keras.layers.Dense(1, activation='sigmoid')(layers)

    nmodel = keras.Model(inputs, logits)
    nmodel.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
    return nmodel


if __name__ == '__main__':
    # hyperparameters
    NUM = 10000
    DIM = 16
    MAX = 100

    # input & preprocess data
    data_path = "tmp/sarcasm/Sarcasm_Headlines_Dataset.json"
    sentences, sarcasm = input_fn(data_path, train_split=.8)
    x_train, x_valid, w_idx = preprocess(sentences, word_num=NUM, seq_length=MAX)
    y_train, y_valid = sarcasm

    # fit model
    model = architecture(MAX, NUM, DIM)
    model.summary()
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=10)
