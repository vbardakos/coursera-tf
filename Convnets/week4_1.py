import os
import zipfile
import requests
from tensorflow import keras
from PIL import UnidentifiedImageError


def downloader(url: str, path: str, ignore=False):
    if os.path.isdir(path):
        path = os.path.join(path, url.split('/')[-1])
    elif not path:
        path = url.split('/')[-1]

    if ignore or not os.path.exists(path):
        response = requests.get(url, verify=False)
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        print('Warning : File already exists - try `ignore` param.')
    return path


def unzip_data(path: str, replace=False):
    if os.path.isfile(path):
        my_dir = ''.join(path.split('.')[:-1])
        if replace and os.path.exists(my_dir):
            os.rmdir(my_dir)
        elif not os.path.exists(my_dir):
            os.mkdir(my_dir)
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(my_dir)
        else:
            print('File already Exists - try `replace`')


def data_integrity_check(path: str):
    counter = list()
    for p, _, file_list in os.walk(path):
        if file_list:
            for f in file_list:
                file_path = os.path.join(p, f)
                if os.path.getsize(file_path):
                    try:
                        keras.preprocessing.image.load_img(file_path)
                    except UnidentifiedImageError:
                        os.remove(file_path)
                        counter.append(file_path)
                        print(f"{file_path} got deleted")
                else:
                    os.remove(file_path)
                    counter.append(file_path)
                    print(f"{file_path} got deleted")
    if counter:
        print(f"Total Deleted : {len(file_path)} files")
    else:
        print('Nothing deleted')

    return counter


if __name__ == '__main__':
    # download data
    train_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
    valid_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'
    test_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-validation.zip'

    f_paths = list()
    for urls in (train_url, valid_url, test_url):
        f_paths.append(downloader(urls, 'tmp'))

    # unzip data
    for zipped in f_paths:
        unzip_data(zipped)

    # check data integrity
    data_integrity_check('tmp/rps')
    data_integrity_check('tmp/rps-test-set')
    data_integrity_check('tmp/rps-validation')
