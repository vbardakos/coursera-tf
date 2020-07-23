import os
import shutil
import zipfile
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_file(path: str, new_path: str = ''):
    f_name = os.path.join(new_path, os.path.split(path)[1])

    if os.path.isfile(path):
        if os.path.exists(f_name):
            print('File already exists')
        else:
            try:
                os.mkdir(new_path)
            except OSError:
                pass
            shutil.move(path, f_name)
    else:
        print("File doesn't exist or already moved")

    return f_name


def unzip_data(path: str, rename: str = '', delete_zip: bool = False):
    if rename:
        new_dir = os.path.join(os.path.split(path)[0], rename)
    else:
        new_dir = os.path.join(os.path.split(path)[0], os.path.split(path)[1].split('.')[0])

    if os.path.isfile(path) and not os.path.isdir(new_dir):
        with zipfile.ZipFile(path) as f:
            f.extractall(new_dir)
        if delete_zip:
            os.remove(path)
    elif os.path.isdir(new_dir):
        print("New dir already exists")
    else:
        print("Path is not a file")

    return new_dir


class Corpus(object):

    def __init__(self, txt, lab, tkn):
        self.txt = txt
        self.lab = lab
        self.tkn = tkn

    @classmethod
    def input_data(cls, path):
        text, labels = cls.json_reader(path)
        tkn = Tokenizer(oov_token='<OOV>')
        tkn.fit_on_texts(text)
        return cls(text, labels, tkn)

    def word_index(self):
        return self.tkn.word_index

    def padding(self, text: (list, tuple) = ()):
        if text:
            self.txt = text
        seq = self.tkn.texts_to_sequences(self.txt)
        pad = pad_sequences(seq, padding='post')
        return seq, pad

    @staticmethod
    def json_reader(path):
        my_data = np.empty(0)
        for line in open(path):
            my_data = np.append(my_data, json.loads(line))

        header = np.vectorize(lambda x: x['headline'])(my_data)
        sarcasm = np.vectorize(lambda x: x['is_sarcastic'])(my_data)
        return header, sarcasm


if __name__ == '__main__':
    # get & extract data
    my_dir = get_file('C:/Users/grate/Downloads/30764_533474_compressed_Sarcasm_Headlines_Dataset.json.zip', 'tmp')
    my_dir = unzip_data(my_dir, 'sarcasm', delete_zip=True)

    # file path
    my_dir = os.path.join(my_dir, 'Sarcasm_Headlines_Dataset.json')

    # corpus & padding
    corpus = Corpus.input_data(my_dir)
    print(corpus.word_index())
    print(corpus.padding())

    # try keras.utils.getfile with this link after
    'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
