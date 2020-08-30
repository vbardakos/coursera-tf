from typing import Tuple, List, Dict

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def input_fn(ds_name: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    ds, info = tfds.load(name=ds_name, with_info=True, as_supervised=True)
    return ds['train'], ds['test'], info


def preprocess(dataset: tf.data.Dataset) -> Tuple[str, int]:
    def f(x):
        return x[0].numpy().decode('utf8'), x[1].numpy()

    arr = np.array(list(map(f, dataset)))
    return arr[..., 0], arr[..., 1].astype(int)


def tokenize(train_set: np.ndarray, test_set: np.ndarray, max_len: int,
             truncate: str = 'post', **tkn_kwargs) -> (List[np.ndarray], Dict[str, int]):
    tkn = Tokenizer(**tkn_kwargs)
    tkn.fit_on_texts(train_set)

    padded_sequences = list()
    for features in [train_set, test_set]:
        sequence = tkn.texts_to_sequences(features)
        padded_sequences.append(pad_sequences(sequence, maxlen=max_len, truncating=truncate))

    return padded_sequences, tkn.word_index


def model_fn(vocabulary: int, dims: int, max_length: int):
    inputs = keras.Input(shape=[max_length], dtype='int64')
    layers = keras.layers.Embedding(vocabulary, dims)(inputs)
    layers = keras.layers.Flatten()(layers)
    layers = keras.layers.Dense(64, activation='relu')(layers)
    logits = keras.layers.Dense(1, activation='sigmoid')(layers)

    nmodel = keras.Model(inputs, logits)
    nmodel.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])
    return nmodel


def embedding_to_projector(trained_model: keras.Model, word_index: Dict[str, int]):
    """
    Writes 'vectors.tsv' and 'meta.tsv' in a compatible format for use with
    "http://projector.teensorflow.org/"

    :param trained_model: trained keras Model (not history)
    :param word_index: word_index from the tokenizer
    :return: None
    """
    word_index = {val: key for key, val in word_index.items()}

    for layer in trained_model.layers:
        if isinstance(layer, keras.layers.Embedding):
            weights = layer.get_weights()[0]
            break
    else:
        raise Exception('Embedding Layer does not exist')

    with open("vectors.tsv", 'w', encoding='utf-8') as vec, open("meta.tsv", 'w', encoding='utf-8') as meta:
        for idx, weight in enumerate(weights):
            vec.write('\t'.join([str(x) for x in weight]) + '\n')
            meta.write(word_index[idx + 1] + '\n')


if __name__ == '__main__':
    # input data
    train, valid, _ = input_fn('imdb_reviews')

    # preprocess
    x_train, y_train = preprocess(train)
    x_valid, y_valid = preprocess(valid)

    # tokenize
    TKN_KWARGS = {'num_words': 10000,
                  'oov_token': '<OOV>'}
    (x_train, x_valid), word_idx = tokenize(x_train, x_valid, max_len=120, truncate='post', **TKN_KWARGS)

    # train model
    my_model = model_fn(10000, 16, 120)
    my_model.fit(x_train, y_train,
                 validation_data=(x_valid, y_valid),
                 epochs=3)

    # write weights/words
    embedding_to_projector(my_model, word_idx)
