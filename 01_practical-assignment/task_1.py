#!/usr/bin/env python
from __future__ import print_function, division

from utils import get_data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances


class T1(object):
    def __init__(self, x_train, y_train, x_test, y_test, alternative_dst, metric):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alternative_dst = alternative_dst
        self.metric = metric
        self.train_digit = []
        self.train_cardinality = []
        self.test_digit = []
        self.test_cardinality = []
        self.label_index_list = []
        self.centers = []
        self.dst_radius = np.zeros(10)
        self.dst_centers = np.zeros((10, 10))

    def explore_data(self, console=False, plot=False):
        self.train_digit, self.train_cardinality = np.unique(self.y_train, return_counts=True)
        self.test_digit, self.test_cardinality = np.unique(self.y_test, return_counts=True)
        if console:
            print("Training set:", dict(zip(self.train_digit.astype(int), self.train_cardinality)))
            print("Testing set:", dict(zip(self.test_digit.astype(int), self.test_cardinality)))
            print("x_train.shape, y_train.shape", self.x_train.shape, self.y_train.shape)
            print("x_test.shape, y_test.shape", self.x_test.shape, self.y_test.shape)
            print("train_digit, train_cardinality", self.train_digit, self.train_cardinality)
            print("test_digit, test_cardinality", self.test_digit, self.test_cardinality)
        if plot:
            plt.rcParams['figure.figsize'] = [10, 5]
            _ = plt.bar(self.train_digit, self.train_cardinality)
            _ = plt.bar(self.test_digit, self.test_cardinality)
            _ = plt.xticks(np.arange(10))
            _ = plt.legend(("train", "test"))
            plt.show()
        return self.train_digit, self.train_cardinality, self.test_digit, self.test_cardinality

    def indexes_list(self):
        df_y = pd.DataFrame(self.y_train, columns=['value'])
        for i in range(10):
            self.label_index_list.append(df_y.index[df_y['value'] == i].tolist())
        return self.label_index_list

    def calc_centers(self):
        for d in range(10):
            accum = np.zeros((256,))
            for i in self.label_index_list[d]:
                accum += self.x_train[i]
            self.centers.append(accum / len(self.label_index_list[d]))

    def calc_radius(self, console=False, plot=False):
        for i in range(10):
            c = self.centers[i]
            for index in self.label_index_list[i]:
                if self.alternative_dst:
                    e_dst = pairwise_distances(c.reshape(16, 16), self.x_train[index].reshape(16, 16), self.metric)
                    if self.dst_radius[i] < np.amax(e_dst):
                        self.dst_radius[i] = np.amax(e_dst)
                else:
                    e_dst = np.linalg.norm(c - self.x_train[index])
                    if self.dst_radius[i] < e_dst:
                        self.dst_radius[i] = e_dst
            if console:
                print("i=", i, " >> radius: ", self.dst_radius[i,])
        if plot:
            plt.rcParams['figure.figsize'] = [10, 5]
            _ = plt.bar(self.train_digit, self.dst_radius)
            _ = plt.xticks(np.arange(10))
            _ = plt.legend("radius")
            plt.show()

    def calc_dist(self, console=False, plot=False):
        for i in range(10):
            c = self.centers[i]
            for index in range(10):
                if self.alternative_dst:
                    e_dst = pairwise_distances(c.reshape(16, 16), self.centers[index].reshape(16, 16), self.metric)
                    self.dst_centers[i][index] = np.amax(e_dst)
                else:
                    e_dst = np.linalg.norm(c - self.centers[index])
                    self.dst_centers[i][index] = e_dst
            if console:
                res = dict(zip(self.train_digit, self.dst_centers[i,]))
                print("i=", i, " >> dst between the centers: ", res)
                del res[i]
                closest = min(zip(res.values(), res.keys()))
                print("i=", i, " >> closest to center of: ", closest[1], " (", closest[0], ")")
            if plot:
                plt.rcParams['figure.figsize'] = [10, 5]
                _ = plt.bar(self.train_digit, self.dst_centers[i,])
                _ = plt.xticks(np.arange(10))
                _ = plt.title("i={}".format(i))
                _ = plt.legend("dst")
                plt.show()


def run_1(x_train, y_train, x_test, y_test, alternative_dst=False, metric="euclidean", console=False, plot=False):
    # create object
    obj = T1(x_train, y_train, x_test, y_test, alternative_dst, metric)
    # explore given data
    obj.explore_data(console, plot)
    # find the indices for each different set of labels
    obj.indexes_list()
    # calculate centers
    obj.calc_centers()
    # calculate radius, visualize
    obj.calc_radius(console, plot)
    # calculate distances between the centers of the centroids
    obj.calc_dist(console, plot)
    return obj


if __name__ == '__main__':
    print("This only executes when this file is executed rather than imported")
    # get data
    online = False
    x_train, y_train = get_data(data="train", online=online, limit=None)
    x_test, y_test = get_data(data="test", online=online, limit=None)
    # run
    task_1 = run_1(x_train, y_train, x_test, y_test, alternative_dst=False, metric="euclidean", console=True, plot=True)
    # TODO wrong results with
    # task_1 = run_1(x_train, y_train, x_test, y_test, alternative_dst=True, metric="euclidean", console=True, plot=True)
