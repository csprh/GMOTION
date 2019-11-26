#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import pandas, math, pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras import backend as be
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from series_to_supervised import series_to_supervised
from visualiseForecast import visualiseForecast
from plotConfidentInt import plot_mean_and_CI, plotConfidentInt
from reset_keras import reset_keras

useGps = False
if useGps:
    ################# LOAD GPS data #############################
    datafiles = ['LEED_east', 'LEED_LOS', 'LEED_vert', 'LEED_north']
    #datafiles = ['HOOB_east', 'HOOB_LOS', 'HOOB_vert', 'HOOB_north']
    signals = ['east', 'LOS', 'vert', 'north'];    s = 1;
    datafile = datafiles[s]
    dataset = pandas.read_csv(datafile, header=None, usecols=[0,1], delim_whitespace=True, engine='python')   # select date,y(m)
else:
    ################# LOAD InSAR data #############################
    #datafiles = ['DARE_interp_may15-dec18_unfiltered.txt', 'DARE_interp_may15-dec18_Filt.txt', 'DARE_interp_may15-dec18_APS.txt', 'DARE_interp_may15-dec18_TSmooth.txt'];
    datafiles = ['Normanton_interp1day_may15-dec18_unfiltered.txt', 'Normanton_interp1day_may15-dec18_Filt.txt', 'Normanton_interp1day_may15-dec18_APS.txt', 'Normanton_interp1day_may15-dec18_TSmooth.txt'];
    #datafiles = ['Normanton_interp6day_may15-dec18_unfiltered.txt', 'Normanton_interp6day_may15-dec18_Filt.txt', 'Normanton_interp6day_may15-dec18_APS.txt', 'Normanton_interp6day_may15-dec18_TSmooth.txt'];
    #datafiles = ['Normanton_orig_may15-dec18_unfiltered.txt', 'Normanton_orig_may15-dec18_Filt.txt', 'Normanton_orig_may15-dec18_APS.txt', 'Normanton_orig_may15-dec18_TSmooth.txt'];
    ##datafiles = ['Leeds_interp1day_may15-dec18_unfiltered.txt', 'Leeds_interp1day_may15-dec18_Filt.txt', 'Leeds_interp1day_may15-dec18_APS.txt', 'Leeds_interp1day_may15-dec18_TSmooth.txt'];
    filterlevels = ['unfiltered', 'Filt', 'APS', 'TSmooth'];     fl = 3
    datafile = datafiles[fl]
    ## locations with highest seasonality: 4062,4058 for Normanton; 12994 for Leeds
    dataset = pandas.read_csv(datafile, header=None, usecols=[4062], engine='python')

#### show data
dataset, datafile
import pudb; pu.db
# set displacement values to milimeters
values = dataset.values[:,1]*1000 if useGps else dataset.values
print(values.shape)

# plot displacement values in milimeters
plt.plot(values)
ndates = len(values)
plt.xlabel('Time blocks of %i dates per location' % (ndates))
plt.ylabel('Displacement (mm)')
plt.title('Deformation MAY\'15 - DEC\'18')
#plt.show()

# select range of dates for years between MAY 2015 and DEC 2018
ndates = len(values)
useVal = False; earlySeason = False; plotFit = False; maxPredDays = 1
daysObs = 4#30          # set to 30 or 4 with or without interpolation, respectively
test_size = 60#365      # set to 365 or 60 with or without interpolation, respectively
nMonths = int(ndates/daysObs)+1   # number of total months in the data
print("%i total months" % (nMonths))
months = 31          # number of previous months to learn from
predMonths = 1       # number of months to forecast
seed = 4             # number of random seeds

# specify columns (locations) to consider and to plot
groups = range(1)
# plot each location
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, group+1)
    plt.plot(values[group*ndates:(group+1)*ndates])
    plt.title('Displacements for location '+str(groups.index(group)+1))
#plt.show()

# split data into groups of locations given the dates and total size of the considered dataset
groups = range(0, int(len(dataset)/ndates))
values = np.delete(values, range(len(groups)*ndates,len(values)), 0)      # delete end locations
values = np.transpose(np.array(values, ndmin=2)) if useGps else np.array(values, ndmin=2)
print(values.shape)
# ensure all data is float
values = values.astype('float32')
nfeatures = 1       # number of different type of features (1: displacements only)
feat = 0;

# normalize Cumulative Displacements by location group
scaled = np.zeros(shape=(values.shape[0], nfeatures))
scalerCD = MinMaxScaler(feature_range=(0, 1))
for group in groups:
    scaled[group*ndates:(group+1)*ndates,0] = np.transpose(np.array(scalerCD.fit_transform(values[group*ndates:(group+1)*ndates]), ndmin=2))
#scaled = np.transpose(scaled)
print(scaled.shape)
np.min(scaled[:,0]),np.max(scaled[:,0])   # check data is normalised within range [0,1]

#PRH There seems to be lots of variables unused in this version of the code.
#PRH I think it would make the code more readable if these were rationalised
tr = values[:-test_size].reshape(1,ndates-test_size)[0,:]
y_te = values[-test_size:].reshape(1,test_size)[0,:]
#PRH Have you considered making your loops into functions.  I find this number of lines in one overall function hard to read
#PRH The inner loop would be better and more readable as a function (this may be difficult, but I think it would be more readable)
#PRM Although it may make sense to characterise signals over different periods of time I don't think anything less than a year makes sense
#PRH Also, the look_back and look_forward variables are always equal.  Wouldn't it make more sense to have more days to predict from and less to predict in some circumstances?
#PRH Lastly, predicting for significantly less than 6 months won't it just be predicting noise and / or trend wouldn't it?

for pred in range(predMonths, 16):
    errs, preds = [], []
    for seed in range(4,7): #print(seed)
        plt.close()
        # run experiments - learning and test for each month
        np.random.seed(seed)
        # Save Scores
        rmse, var, mae, mdae,  r2 =  [], [], [], [], []
        trainPredict, testPredict = [], []
        if useVal: valPredict = []
        multistep = True;
        for month in range(months-10,months):    # maximum period of 3 years observed
            print("Learning from previous %i months and %i monthly observations with random seed %i ..." % (month+1, pred, seed))

            # define temporal windows of observations
            #PRH is this right? pred is never used in this loop, shouldn't the two lines below use pred rather than prefMonths?
            look_back = daysObs*predMonths
            look_forward = daysObs*predMonths

            # frame as multivar supervised learning for each location
            train = series_to_supervised(scaled[:-test_size], ndates-test_size, nfeatures, look_back, look_forward)
            #PRH...The use of series_to_supervised for the test set is confusing.  I would not use it
            test = series_to_supervised(scaled[-test_size:], test_size, nfeatures, 0, look_forward)
            # drop columns we don't want to predict
            print(train.head(),test.head())
            print(train.shape, test.shape)

            n_obs = look_back * nfeatures
            n_futObs = look_forward * nfeatures

            # split into train and test sets
            train = train.values   # scaled[:-test_size]
            test = test.values     # scaled[-test_size:]
            #print(train.shape,test.shape)

            # define number of training samples
            n_samples = (month+1)*daysObs

            # Multiple Lag Timesteps
            ############################################################################################################################################
            ########################## https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ ############################
            ############################################################################################################################################
            # This will modify the LSTM' input and output shapes in [samples, observations, features times future observations]
            if multistep:
                #PRH I think this is a confusing way of defining the testset.  I think it should simply be defined using the "scaled" variable
                test_X = train[-1, -n_futObs:]
                test_y = test[0, :n_futObs]
                if not useVal:
                    val_X, val_y = np.empty(0), np.empty(0)
                    train_X = train[-n_samples:, :n_obs]                                # training set consists of the last samples and chosen number of observations
                    train_X = train_X.reshape((train_X.shape[0], n_obs, nfeatures))
                    train_y = train[-n_samples:, -n_futObs:]                            # training sequence for testing consists of the last observation from the first training sample
                else:
                    #test_X, test_y = test[:n_futObs, :n_futObs], test[0, :n_futObs]    # test set consists of future observations from the next months; test sequence consists of the future observations of the first test sample
                    #test_X = test_X.reshape((test_X.shape[0], look_forward, nfeatures))
                    if earlySeason:
                        # there can't be more test samples to be predicted than number of training samples before the previous season
                        # this needs to be compensated by the concatenation with training samples from the previous season after the validation set
                        maxPredDays = np.abs(train.shape[0]-test_size-n_samples)+1 if train.shape[0]-test_size-n_samples < 0 else 1

                        # Set the starting of the last season in the validation data and the end of it still as part of the training data.
                        # concatenation to be equally sized for validation and test sets
                        if -test_size+n_obs+maxPredDays < 0:
                            tr = np.concatenate((train[-test_size-n_samples:-test_size-1, :n_obs], train[-test_size+n_obs:-test_size+n_obs+maxPredDays, -n_obs:].reshape(-test_size-n_samples, n_obs)))
                            train_y = np.concatenate((train[-test_size-n_samples:-test_size-1, -nfeatures], train[-test_size+n_obs:-test_size+n_obs+maxPredDays, -nfeatures]))
                        else:
                            tr = np.concatenate((train[-test_size-n_samples:-test_size-1, :n_obs], train[-test_size+n_obs:, -n_obs:].reshape(-test_size-n_samples, n_obs)))
                            train_y = np.concatenate((train[-test_size-n_samples:-test_size-1, -nfeatures], train[-test_size+n_obs:, -nfeatures]))
                        train_X = tr   # training set consists of past observations from the previous months optimised for one (-n_obs) or last (-nfeatures) observation
                        val_X, val_y = train[-test_size:-test_size+n_futObs, -n_futObs:], train[-test_size+n_futObs, -n_futObs:]  # validation set consists of future observations from the previous months; validation test sequence consists of the future observations of the last sample
                    else:
                        # late stages of the season
                        train_X, train_y = train[-n_samples:, :n_obs], train[-n_samples:, -n_obs]       # training set consists of past observations from the previous months optimised for one (-n_obs) or last (-nfeatures) observation
                        val_X, val_y = train[-n_futObs:, -n_futObs:], train[-1, -n_futObs:]             # validation set consists of future observations from the previous months; test sequence consists of the future observations of the last sample
                    train_X = train_X.reshape((train_X.shape[0], n_obs, nfeatures))     # training and test sets must have the same size
                    val_X = val_X.reshape((val_X.shape[0], n_futObs, nfeatures))        # we ensure this by considering the same number of past and future observations in the training and validation sets, respectively
            print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            be.tensorflow_backend.set_session(tf.Session(config=config))

            # design network
            #model = Sequential()
            #model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))    # 50 neurons, n_obs and nfeatures timesteps as inputs
            #model.add(Dense(n_futObs))                            # n_futObs timesteps as outputs
            #model.compile(loss='mae', optimizer='adam')
            #PRH Have you tried to define the learning rate and it's variation (log decrease for instance)
            #PRH I would try more than one layer of LSTM cells.  Also I would try some drop out layers (after all the LSTM layers or in between them).
            #PRH I would also try to output sequences from the LSTM layers (using return_sequences=True).
            #PRH Maybe something like below:
            #PRH I think it would at somepoint be good to investigate whether to retain the state (i.e. stateful=True) and make the LSTM
            #PRM birectional, batch normalisation, early stopping etc etc.

            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=False))
            model.add(Dropout(0.5))
            model.add(Dense(n_futObs))                            # n_futObs timesteps as outputs
            model.compile(loss='mae', optimizer='adam')
            # fit network
            if useVal:
                history = model.fit(train_X, train_y, epochs=n_samples*10, batch_size=50, validation_data=(val_X, val_y), verbose=1, shuffle=False)
            else:
                history = model.fit(train_X, train_y, epochs=n_samples*10, batch_size=50, verbose=1, shuffle=False)

            # show learning process
            #plt.close()
            #plt.plot(history.history['loss'], label='train')
            #plt.plot(history.history['val_loss'], label='test')
            #plt.xlabel('Epoch')
            #plt.ylabel('Loss')
            #plt.legend()
            #plt.show()

            # Make prediction and invert scaling for forecast
            yhat = model.predict(test_X.reshape((1, n_futObs, nfeatures)))
            inv_yhat = scalerCD.inverse_transform(yhat)[0,:]
            # invert scaling for actual
            inv_y = scalerCD.inverse_transform(test_y.reshape(1,n_futObs))[0,:]

            # Calculate regression scores for all metrics and future observations
            #PRH I'm not sure that comparing all of these different metrics is aiding the analysis
            #PRH I think you should just stick to MSE
            rmse.append(math.sqrt(mean_squared_error(inv_y, inv_yhat)))
            var.append(explained_variance_score(inv_y, inv_yhat))
            mae.append(mean_absolute_error(inv_y, inv_yhat))
            mdae.append(median_absolute_error(inv_y, inv_yhat))
            r2.append(r2_score(inv_y, inv_yhat))

            # save predictions
            trainPredict.append(scalerCD.inverse_transform(model.predict(train_X))[:,0])
            testPredict.append(inv_yhat)

            del model, history
            be.clear_session(); reset_keras()   # Reset Keras Session

            # plot forecasting
            plt.close()
            plt.plot(values, label='Real Sequence', color='blue')
            plt.plot(np.concatenate((np.full(ndates-test_size, np.nan), inv_yhat)), label='Forecast-'+str(month+1), color='green')
            plt.xlabel('Day')                          # use for the averaged CDs
            plt.ylabel('Cumulative Displacement')
            plt.legend(bbox_to_anchor=(0.9, 1))
            #plt.show()

            plt.close()
            plt.plot(inv_y, label='Real Sequence')
            plt.plot(inv_yhat, label='Forecast-'+str(month+1))
            plt.legend(loc='best')
            #plt.show()
            print('Test RMSE: %.3f' % math.sqrt(mean_squared_error(inv_y, inv_yhat)))

        #visualiseForecast(plotFit, earlySeason, test_size, values, ndates, months, daysObs, n_futObs, trainPredict, testPredict, rmse)

        # generate 3 sets of random means and confidence intervals to plot
        errs.append(rmse); preds.append(inv_yhat)

    pickle.dump(errs, open('errs'+str(pred)+'.pkl', 'wb'))
    pickle.dump(preds, open('preds'+str(pred)+'.pkl', 'wb'))

#pred = 1
#errs = pickle.load(open('errs'+str(pred)+'.pkl', 'rb'))
#preds = pickle.load(open('preds'+str(pred)+'.pkl', 'rb'))
#np.argmin(errs[0]),np.argmin(errs[1]),np.argmin(errs[2])
#idx = 11
#start = ndates-test_size-daysObs*(idx+1)-n_obs-n_futObs
#x_tr = values[start:-test_size-len(inv_y)].reshape(1,ndates-test_size-start-len(inv_y))[0,:]
#minErrs = [errs[0][idx],errs[1][idx],errs[2][idx]]
#plotConfidentInt(minErrs, preds, inv_y, y_te, tr, x_tr, idx+1, 'all')
#plotConfidentInt(minErrs, preds, inv_y, y_te, tr, x_tr, idx+1, 'zoom')
#plotConfidentInt(minErrs, preds, inv_y, y_te, tr, x_tr, idx+1, 'super')
#np.mean(minErrs),np.std(minErrs)
