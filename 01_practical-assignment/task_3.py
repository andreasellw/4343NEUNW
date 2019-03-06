#!/usr/bin/env python
from __future__ import print_function, division
from future.utils import iteritems

from utils import get_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import multivariate_normal as mvn


class T3(object):
    def __init__(self, naive=False):
        self.naive = naive
        self.gaussians = dict()
        self.priors = dict()

    def fit(self, x, y, smoothing=1e-2):
        n, d = x.shape
        labels = set(y)
        for c in labels:
            current_x = x[y == c]
            if self.naive:
                self.gaussians[c] = {
                    'mean': current_x.mean(axis=0),
                    'var': current_x.var(axis=0) + smoothing,
                }
            else:
                self.gaussians[c] = {
                    'mean': current_x.mean(axis=0),
                    'cov': np.cov(current_x.T) + np.eye(d) * smoothing,
                }
            self.priors[c] = float(len(y[y == c])) / len(y)

    def score(self, x, y):
        p = self.predict(x)
        return np.mean(p == y)

    def predict(self, x):
        n, d = x.shape
        k = len(self.gaussians)
        p = np.zeros((n, k))
        if self.naive:
            for c, g in iteritems(self.gaussians):
                mean, var = g['mean'], g['var']
                p[:, c] = mvn.logpdf(x, mean=mean, cov=var) + np.log(self.priors[c])
        else:
            for c, g in iteritems(self.gaussians):
                mean, cov = g['mean'], g['cov']
                p[:, c] = mvn.logpdf(x, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(p, axis=1)

    def plot(self, title):
        print("Plotting mean of each class...")
        for c, g in iteritems(self.gaussians):
            plt.imshow(g['mean'].reshape(16, 16))
            plt.title("{} - {}".format(title, c))
            plt.show()


def run(x_train, y_train, x_test, y_test, plot=False, naive=False):
    print("Bayes(naive={})".format(naive))
    # init model
    model = T3(naive)
    # fit model
    time = datetime.now()
    model.fit(x_train, y_train)
    print("Training time:", (datetime.now() - time))

    time = datetime.now()
    print("Train accuracy:", model.score(x_train, y_train))
    print("Time to compute train accuracy:", (datetime.now() - time), "Train size:", len(y_train))

    time = datetime.now()
    print("Test accuracy:", model.score(x_test, y_test))
    print("Time to compute test accuracy:", (datetime.now() - time), "Test size:", len(y_test))
    if plot:
        model.plot("Bayes(naive={})".format(naive))
    return model


if __name__ == '__main__':
    print("This only executes when this file is executed rather than imported")
    # get data
    online = False
    x_train, y_train = get_data(data="train", online=online, limit=None)
    x_test, y_test = get_data(data="test", online=online, limit=None)
    # run
    print("\n")
    model_bayes = run(x_train, y_train, x_test, y_test, plot=False, naive=False)
    print("\n")
    model_naive_bayes= run(x_train, y_train, x_test, y_test, plot=False, naive=True)
