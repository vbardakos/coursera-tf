"""
Contents:
    download : Downloads zip file with our data from https
    unzip_data : unzips our data
    input_fn : Loads data using an image generator
"""
# download & input_fn
import os
import requests
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ConvNet
from tensorflow import keras
import numpy as np


def download(url: str, path: str = None, file_name: str = None,
             replace_file=False, verify_url=True):
    """
    downloads url

    :param url: url to download
    :param path: destination path. None points at current directory
    :param file_name: name saved. None keeps original name
    :param replace_file: If True replaces files with the same name
    :param verify_url: Ignores secure communication certificate
    :return: local path of the file
    """
    if not file_name:
        file_name = url.split('/')[-1]

    if not path:
        path = os.getcwd().replace('\\', '/')
        full_path = f"{path}/{file_name}"
    else:
        path = path.replace('\\', '/')
        if path[-1] == '/':
            full_path = path + file_name
        else:
            full_path = f"{path}/{file_name}"

    if replace_file or not os.path.exists(full_path):
        if not os.path.exists(path):
            os.mkdir(path)
        print("Download starts...")
        file = requests.get(url, verify=verify_url)
        with open(full_path, 'wb') as f:
            f.write(file.content)
    else:
        print(f"Warning : File <{file_name}> has already been downloaded.")

    return full_path


def unzip_data(file_path: str, replace=False):
    """
    unzips a file

    :param file_path: file path to unzip
    :param replace: deletes directory with the same name
    :return: unzips file and creates directory with the same name
    """
    assert os.path.isfile(file_path), "Path does not contain a file"
    path_list = file_path.split('.')

    if replace:
        os.rmdir(file_path)
    elif os.path.exists(str().join(path_list[:-1])):
        pass
    elif path_list[-1] == "zip":
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(file_path.split('.')[0])
    else:
        raise Exception('File path points to a file')

    file_path = str().join(path_list[:-1])
    return file_path


def input_fn(file_path: str, batch: int):
    """
    Returns a keras image generator from a path
    :param file_path: Given file path
    :param batch: assigns batch size
    """
    generator = ImageDataGenerator(rescale=1/255.)
    generator = generator.flow_from_directory(file_path,
                                              target_size=(300, 300),
                                              batch_size=batch,
                                              class_mode='binary')
    return generator


class AccCallBack(keras.callbacks.Callback):

    def __init__(self, patience=10, acc_rate=.9):
        super(AccCallBack, self).__init__()
        self.patience = patience
        self.acc = acc_rate
        self.wait = 0  # compared to patience
        self.best_loss = np.inf  # holds best recorded loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if logs.get('acc') < self.acc and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        elif current_loss > self.best_loss:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nLossCallback : Loss started increasing at {self.wait} with patience {self.patience}" * 3)
                self.model.stop_training = True
        else:
            print(f"\nAccuracyCallback : Model reached {self.acc * 100}% accuracy" * 3)
            self.model.stop_training = True


if __name__ == '__main__':
    # download files
    link_train = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
    path_train = download(link_train, path="tmp/", verify_url=False)
    link_test = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
    path_test = download(link_test, path="tmp/", verify_url=False)

    # unzip them & get directory
    path_train = unzip_data(path_train)
    path_test = unzip_data(path_test)

    # input
    gen_train = input_fn(path_train, batch=128)
    gen_test = input_fn(path_test, batch=32)