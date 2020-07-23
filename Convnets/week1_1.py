"""
Contents:
"""
import os
import re
import shutil
from tqdm import tqdm

from Introduction.week4 import unzip_data


def name_to_label(path, regex: str, labels: (list, tuple) = None, to_path: str = None):
    """
    labels a file list with regex and move them to labeled directories

    :param path: files' path
    :param regex: regex rule to get the labels
    :param labels: acceptable labels (should be lowered)
    :param to_path: destination path
    :return: unlabeled data
    """
    os.chdir(path)
    unlabeled = list()

    if not to_path:
        to_path = ".."

    def move_to_label(f, l_path):
        try:
            os.mkdir(l_path)
            shutil.move(f, l_path)
        except OSError:
            shutil.move(f, l_path)

    for file in tqdm(os.listdir()):
        label = re.findall(regex, file)[0].lower()
        if labels:
            if label in labels:
                move_to_label(file, f"{to_path}/{label}")
            else:
                unlabeled.append(file)
        else:
            move_to_label(file, f"{to_path}/{label}")

    if unlabeled:
        print('Warning : returns unlabeled files:', unlabeled, sep='\n')
    else:
        print('Everything is labeled')
        os.chdir("..")
        os.rmdir(path.split('/')[-1])

    return unlabeled


if __name__ == '__main__':

    # unzip train data
    unzip_data('tmp/dogs-vs-cats.zip')
    unzip_data('tmp/dogs-vs-cats/train.zip')

    # unzip validation data
    unzip_data('tmp/dogs-vs-cats/test1.zip')

    # labeled folders
    train_path = "tmp/dogs-vs-cats/train/train"
    name_to_label(path=train_path, regex="((?i)dog|(?i)cat)", labels=('dog', 'cat'))
