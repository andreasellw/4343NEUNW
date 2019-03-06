#!/usr/bin/env python
from __future__ import print_function, division

from utils import get_data
from task_1 import run_1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, pairwise_distances

import itertools


class T2(object):
    def __init__(self, x_train, y_train, x_test, y_test, alternative_dst, metric):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alternative_dst = alternative_dst
        self.metric = metric

    def init_t1(self):
        tmp = run_1(self.x_train, self.y_train, self.x_test, self.y_test, alternative_dst=self.alternative_dst,
                    metric=self.metric)
        self.task_1 = tmp

    def neighbors(self, trainSet, testInst):
        dst = np.zeros(10)
        for i in range(10):
            if self.alternative_dst:
                dst[i] = np.amax(pairwise_distances(trainSet[i].reshape(16, 16), testInst.reshape(16, 16), self.metric))
            else:
                dst[i] = np.linalg.norm(trainSet[i] - testInst)
        return dst

    def kNearestNeighbors(self, trainSet, testInst):
        list = self.neighbors(trainSet, testInst)
        min = np.min(list)
        for label in range(10):
            if list[label] == min:
                return label

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()


def run_2(x_train, y_train, x_test, y_test, alternative_dst=False, metric="euclidean"):
    # create object
    obj = T2(x_train, y_train, x_test, y_test, alternative_dst, metric)
    obj.init_t1()
    # train

    data_train = obj.x_train
    pred_train = np.zeros(len(data_train))

    for i in range(len(data_train)):
        pred_train[i] = obj.kNearestNeighbors(obj.task_1.centers, data_train[i,])

    train_y_true = obj.y_train
    train_y_pred = pred_train.astype(int)

    cnf_train = confusion_matrix(train_y_true, train_y_pred)
    print("Train accuracy: {} ({} / {})".format((np.trace(cnf_train) / obj.x_train.shape[0] * 100), np.trace(cnf_train),
                                                len(data_train)))
    print("\n train:\n", cnf_train)
    print("\n\n")

    # test

    data_test = obj.x_test
    pred_test = np.zeros(len(data_test))

    for i in range(len(data_test)):
        pred_test[i] = obj.kNearestNeighbors(obj.task_1.centers, data_test[i,])

    test_y_true = obj.y_test
    test_y_pred = pred_test.astype(int)

    cnf_test = confusion_matrix(test_y_true, test_y_pred)
    print("Test accuracy: {} ({} / {})".format((np.trace(cnf_test) / obj.x_test.shape[0] * 100), np.trace(cnf_test),
                                               len(data_test)))
    print("\n test:\n", cnf_test)
    print("\n\n")

    # plot

    class_names = range(10)
    obj.plot_confusion_matrix(cnf_train, classes=class_names, normalize=False, title="Train")
    obj.plot_confusion_matrix(cnf_test, classes=class_names, normalize=False, title="Test")

    return obj


if __name__ == '__main__':
    print("This only executes when this file is executed rather than imported")
    # get data
    online = False
    x_train, y_train = get_data(data="train", online=online, limit=None)
    x_test, y_test = get_data(data="test", online=online, limit=None)
    # run
    task_2 = run_2(x_train, y_train, x_test, y_test, alternative_dst=False, metric="euclidean")
    # TODO wrong results with
    # task_2 = run_2(x_train, y_train, x_test, y_test, alternative_dst=True, metric="euclidean")
