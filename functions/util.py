import os
import shutil
import urllib
import subprocess
import random
from typing import List

import emoji
import numpy as np
from numpy import logical_and, sum as t_sum


#### CONSTANTS ####
SEM_EVAL_FOLDER = 'SemEval2018-Task3'
TRAIN_FILE = SEM_EVAL_FOLDER + '/datasets/train/SemEval2018-T3-train-taskA_emoji.txt'
TEST_FILE = SEM_EVAL_FOLDER + '/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'


def download_zip(url: str, dir_path: str):
    import zipfile
    if url.endswith('zip'):
        print('Downloading dataset file')
        path_to_zip_file = 'downloaded_file.zip'
        urllib.request.urlretrieve(url, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall('./')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        if os.path.exists(path_to_zip_file):
            os.remove(path_to_zip_file)
        if os.path.exists(dir_path + '-master'):
            shutil.move(dir_path + '-master', dir_path)
        elif os.path.exists(dir_path + '-main'):
            shutil.move(dir_path + '-main', dir_path)
    print(f'Downloaded dataset to {dir_path}')


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def download_irony():
    if os.path.exists(SEM_EVAL_FOLDER):
        return
    else:
        try:
            git('clone', 'https://github.com/Cyvhee/SemEval2018-Task3.git')
            return
        except OSError:
            pass
    print('Downloading dataset')
    download_zip('https://github.com/Cyvhee/SemEval2018-Task3/archive/master.zip', SEM_EVAL_FOLDER)


def read_dataset_file(file_path):
    with open(file_path, 'r', encoding='utf8') as ff:
        rows = [line.strip().split('\t') for line in ff.readlines()[1:]]
        _, labels, texts = zip(*rows)
    clean_texts = [emoji.demojize(tex) for tex in texts]
    unique_labels = sorted(set(labels))
    lab2i = {lab: i for i, lab in enumerate(unique_labels)}
    return clean_texts, labels, lab2i


def load_datasets():
    download_irony()
    train_texts, train_labels, label2i = read_dataset_file(TRAIN_FILE)
    test_texts, test_labels, _ = read_dataset_file(TEST_FILE)
    
    return train_texts, train_labels, test_texts, test_labels, label2i


def split_data(sentences, labels, split=0.8):
    """Splits the training data into training and development sets."""
    data = list(zip(sentences, labels))
    random.shuffle(data)
    sents, labs = zip(*data)
    index = int(len(sents) * split)
    
    return sents[:index], labs[:index], sents[index:], labs[index:]


def accuracy(predicted_labels, true_labels):
    """Accuracy is correct predictions / all predicitons"""
    correct_count = 0
    for pred, label in zip(predicted_labels, true_labels):
        correct_count += int(pred == label)
    
    return correct_count/len(true_labels) if len(true_labels) > 0 else 0.


def precision(predicted_labels, true_labels, which_label=1):
    """Precision is True Positives / All Positives Predictions"""
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(pred_which)
    if denominator:
        return t_sum(logical_and(pred_which, true_which))/denominator
    else:
        return 0.


def recall(predicted_labels, true_labels, which_label=1):
    """Recall is True Positives / All Positive Labels"""
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(true_which)
    if denominator:
        return t_sum(logical_and(pred_which, true_which))/denominator
    else:
        return 0.


def f1_score(predicted_labels, true_labels, which_label=1):
    """F1 score is the harmonic mean of precision and recall"""
    P = precision(predicted_labels, true_labels, which_label=which_label)
    R = recall(predicted_labels, true_labels, which_label=which_label)
    if P and R:
        return 2*P*R/(P+R)
    else:
        return 0.


def avg_f1_score(predicted_labels: List[int], true_labels: List[int], classes: List[int]):
    """
    Calculate the f1-score for each class and return the average of it
    :return: float
    """
    return sum([f1_score(predicted_labels, true_labels, which_label=which_label) for which_label in classes])/len(classes)