#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import pandas, math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from series_to_supervised import series_to_supervised

useGps = False

if useGps:
    ################# LOAD GPS data #############################
    #datafile = 'LEEK.IGS08.txyz2.csv'
    #datafiles = ['LEED_east', 'LEED_LOS', 'LEED_vert', 'LEED_north']
    datafiles = ['HOOB_east', 'HOOB_LOS', 'HOOB_vert', 'HOOB_north']
    signals = ['east', 'LOS', 'vert', 'north']
    s = 1
    datafile = datafiles[s]
    dataset = pandas.read_csv(datafile, header=None, usecols=[0,1], delim_whitespace=True, engine='python')   # select date,y(m)
else:
    ################# LOAD InSAR data #############################
    #datafiles = ['DARE_interp_may15-dec18_unfiltered.txt', 'DARE_interp_may15-dec18_Filt.txt', 'DARE_interp_may15-dec18_APS.txt', 'DARE_interp_may15-dec18_TSmooth.txt'];
    datafiles = ['Normanton_interp1day_may15-dec18_unfiltered.txt', 'Normanton_interp1day_may15-dec18_Filt.txt', 'Normanton_interp1day_may15-dec18_APS.txt', 'Normanton_interp1day_may15-dec18_TSmooth.txt'];
    #datafiles = ['Leeds_interp1day_may15-dec18_unfiltered.txt', 'Leeds_interp1day_may15-dec18_Filt.txt', 'Leeds_interp1day_may15-dec18_APS.txt', 'Leeds_interp1day_may15-dec18_TSmooth.txt'];
    filterlevels = ['unfiltered', 'Filt', 'APS', 'TSmooth']
    fl = 3
    datafile = datafiles[fl]
    # locations with highest seasonality: 4062 for Normanton; 12994 for Leeds
    dataset = pandas.read_csv(datafile, header=None, usecols=[4062], engine='python')

# set displacement values to milimeters
values = dataset.values[:,1]*1000 if useGps else dataset.values
print(values.shape)
import pudb; pu.db

# select from MAY 2015 as in InSAR
#dataset = dataset[3883:]         # samples from 15MAY11
ndates = len(dataset.values)
nMonths = int(ndates/30)+1
print(nMonths)
if useGps:
    dataset.values[:-30*1,0]         # from ~ last month
    dataset.values[:-30*2,0]         # from ~ last 2 months
    dataset.values[-30*nMonths:,0]   # from all months
    dataset.columns = ['date', 'disp']
    dataset.head(5)

# specify columns (locations) to consider and plot
groups = range(1)
# plot each location
groups = range(0, int(len(dataset)/ndates))
values = np.delete(values, range(len(groups)*ndates,len(values)), 0)      # delete end locations
values = np.transpose(np.array(values, ndmin=2)) if useGps else np.array(values, ndmin=2)
print(values.shape)
# ensure all data is float
values = values.astype('float32')
nf = 1       # features (displacements only)
nfeatures = nf;

# normalize Cumulative Displacements by location group
scaled = np.zeros(shape=(values.shape[0], nfeatures))
scalerCD = MinMaxScaler(feature_range=(0, 1))
for group in groups:
    scaled[group*ndates:(group+1)*ndates,0] = np.transpose(np.array(scalerCD.fit_transform(values[group*ndates:(group+1)*ndates]), ndmin=2))
#scaled = np.transpose(scaled)
print(scaled.shape)
np.min(scaled[:,0]),np.max(scaled[:,0])   # check data is normalised within range [0,1]

# run experiments - learning and test for each month
np.random.seed(4)
# Save Scores
timeRMSE, timeVAR, timeMAE, timeMDAE, timeMRE, timeR2 = np.zeros(shape=(31)), np.zeros(shape=(31)), np.zeros(shape=(31)), np.zeros(shape=(31)), np.zeros(shape=(31)), np.zeros(shape=(31))
stepframe = False; multistep = True;
for month in range(0,31):    # maximum period of 3 years observed
    print("Learning from previous %i months ..." % (month+1))

    ### define set of parameters:
    # timesteps for past and future observations of LSTM
    look_back = 30   # montly observations
    look_forward = 1
    n_obs = look_back * nfeatures

    # frame as multivar supervised learning for each location
    reframed = series_to_supervised(scaled, ndates, nf, look_back, look_forward)
    # drop columns we don't want to predict
    print(reframed.head())
    print(reframed.shape)

    # split into train and test sets
    test_size = 365     # 1 year for testing
    if len(reframed.values)-test_size-(month+1)*30 < 0:
        train = reframed.values[:len(reframed.values)-test_size, :]
    else:
        train = reframed.values[len(reframed.values)-test_size-(month+1)*30:len(reframed.values)-test_size, :]
    test = reframed.values[-test_size:, :]

    print(train.shape,test.shape)

    # Multiple Lag Timesteps
    ############################################################################################################################################
    ########################## https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ ############################
    ############################################################################################################################################
    # This will modify the LSTM' input and output shapes in [samples, observations, features times future observations]
    if multistep:
        train_X, train_y = train[:, :n_obs], train[:, -nfeatures*look_forward]
        test_X, test_y = test[:, :n_obs], test[:, -nfeatures*look_forward]
        train_X = train_X.reshape((train_X.shape[0], look_back, nfeatures))
        test_X = test_X.reshape((test_X.shape[0], look_back, nfeatures))
    else:
        loc = 1   # change this to predict over an specific location
        if loc < int(len(dataset)/ndates):
            loc = (int(len(dataset)/ndates)-loc+1)*nf
            train_X, train_y = np.concatenate((train[:, :-loc], train[:, -loc+nf:]), axis=1), train[:, -loc]
            test_X, test_y = np.concatenate((test[:, :-loc], test[:, -loc+nf:]), axis=1), test[:, -loc]
        else:      # This will predict over the last location only (default if there is only 1 location considered)
            train_X, train_y = train[:, :-nf], train[:, -nf]
            test_X, test_y = test[:, :-nf], test[:, -nf]

            if not stepframe:
                # frame the problem as [sample, 1 timestep, features (ALL observations)]
                train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
                test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
            else:
                # use past observations as time steps of the one input feature --> more accurate framing of the problem
                train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
                test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=10, batch_size=50, validation_data=(test_X, test_y), verbose=1, shuffle=False)

    # show learning process

    # make predictions over the first future observations only and invert back to the original scale
    trainPredict = scalerCD.inverse_transform(model.predict(train_X))
    testPredict = scalerCD.inverse_transform(model.predict(test_X))
    trainY = scalerCD.inverse_transform([train_y])
    testY = scalerCD.inverse_transform([test_y])

    # save figure (change the path!)
    #plt.savefig('/home/cosc/vp17941/linux/results/satsense/seasonality/multistep/GPS_'+signals[s]+'/preds_Hoober_'+str(month+1)+'prevmonths-'+signals[s]+'.png')

    # Calculate reg
