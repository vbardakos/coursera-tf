"""
Hello World NLP
"""
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


def input_fn(*args):
    """
    Sample list of Text
    :return:
    """
    return args


def corpus(sentences):
    token = Tokenizer(num_words=100)
    token.fit_on_texts(sentences)

    return token.word_index


if __name__ == '__main__':
    # get text
    texts = input_fn('I love dogs', 'I love cats', 'I love cows!')

    # check corpus
    print(corpus(texts))
