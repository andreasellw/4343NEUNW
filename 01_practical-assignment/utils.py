#!/usr/bin/env python
from __future__ import print_function, division
from builtins import range, input

import numpy as np
import pandas as pd

url_github = 'https://raw.githubusercontent.com/ndrsllwngr/4343NEUNW/master/01_practical-assignment/data/'
url_local = "data/"


def get_data(data="train", online=False, limit=None, filter=None):
    url = url_local
    if online:
        url = url_github
    print("Reading in and transforming [{}] data...".format(data))
    df1 = pd.read_csv((url + data + '_in.csv'), header=None)
    df2 = pd.read_csv((url + data + '_out.csv'), header=None)
    images = df1.values
    labels = df2.values
    data = np.append(labels, images, axis=1)
    #if filter is not None:
        # filtered_data = data
        # for line in data[0]:
        #    if line != number
        #        remove
    np.random.shuffle(data)
    x = data[:, 1:]
    y = data[:, 0]
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y.astype(int)