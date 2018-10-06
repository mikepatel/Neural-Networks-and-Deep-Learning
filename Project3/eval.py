#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import csv
import numpy as np

current_dir = os.getcwd()
RESULT_PATH = current_dir
GT_LIST = [1, 9, 2, 3]


def read_result(result_path, file_name):
    file_path = os.path.join(result_path, file_name)
    pred_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pred = list(map(int, line.split(',')))
            pred_list.append(np.argmax(pred))
    return pred_list


def read_name(result_path, file_name):
    file_path = os.path.join(result_path, file_name)
    name_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            name_list.append(line.strip())
    print('Student Name: {}'.format(name_list))


def eval_accuracy(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    return np.mean((np.equal(gt, pred).astype(int)))


def main():
    read_name(RESULT_PATH, 'name.csv')

    pred = read_result(RESULT_PATH, 'mnist.csv')
    n_test_sample = len(pred)
    gt_list = np.random.choice(10, n_test_sample)
    accuracy = eval_accuracy(gt_list, pred)
    print('[Result] number of test: {}, accuracy: {}'
          .format(n_test_sample, accuracy))


if __name__ == "__main__":
    main()
