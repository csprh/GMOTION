#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""


import pandas
import numpy as np

# convert series to supervised learning

# Inputs:
#   data: original data as 1D signal
#   nf: number of features (default: 1 (location only))
#   n_in: number of input features (past observations)
#   n_out: number of output features (future observations to be predicted)
#   dropnan: flag which indicates whether or not to drop rows with NaN values
def series_to_supervised(data, nf=1, n_in=1, n_out=1, dropnan=True):
    if nf > 1:
        #groups = int(len(data)/data.shape[0])
        groups = data.shape[1]
        n_vars = groups*nf
        new_data = np.zeros(shape=(data.shape[0],nf*groups))
        for i in range(0, groups):   # groups
            #new_data[:,i*nf:i*nf+nf] = data[i*data.shape[0]:(i+1)*data.shape[0],0:nf]
            new_data[:,i*nf:(i*nf)+nf] = data[:,i,:].reshape(data.shape[0],nf)
        df = pandas.DataFrame(new_data)
    else:
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pandas.DataFrame(data.reshape(data.shape[0],n_vars))
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pandas.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg