from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from NLP.week1_1 import input_fn


class Corpus(object):

    def __init__(self, text, token):
        self.txt = text
        self.tkn = token

    @classmethod
    def fit(cls, texts):
        token = Tokenizer(num_words=100, oov_token='<OOV>')
        token.fit_on_texts(texts)
        return cls(texts, token)

    def padding(self, texts=None):
        if texts:
            self.txt = texts
        seq = self.tkn.texts_to_sequences(self.txt)
        pad = pad_sequences(seq, maxlen=5)
        print(seq, pad, sep='\n')


if __name__ == '__main__':
    txt_train = input_fn('I love my dog', 'You love my dog', 'Do you think my dog is amazing?')
    txt_test = input_fn('i really love my dog', 'my dog loves my manatee')
    c = Corpus.fit(txt_train)
    c.padding()
    c.padding(txt_test)




