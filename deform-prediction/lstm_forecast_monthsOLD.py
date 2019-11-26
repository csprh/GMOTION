#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import pandas, math
#import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
#from sklearn.metrics import max_error
#from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from series_to_supervised import series_to_supervised
#from visualiseForecast import visualiseForecast

useGps = False
import pudb;pu.db

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
    #datafiles = ['Normanton_orig_may15-dec18_unfiltered.txt', 'Normanton_orig_may15-dec18_Filt.txt', 'Normanton_orig_may15-dec18_APS.txt', 'Normanton_orig_may15-dec18_TSmooth.txt'];
    ##datafiles = ['Leeds_interp1day_may15-dec18_unfiltered.txt', 'Leeds_interp1day_may15-dec18_Filt.txt', 'Leeds_interp1day_may15-dec18_APS.txt', 'Leeds_interp1day_may15-dec18_TSmooth.txt'];
    filterlevels = ['unfiltered', 'Filt', 'APS', 'TSmooth'];     fl = 3
    datafile = datafiles[fl]
    ## locations with highest seasonality: 4062,4058 for Normanton; 12994 for Leeds
    dataset = pandas.read_csv(datafile, header=None, usecols=[4062], engine='python')

#### show data
dataset, datafile

# set displacement values to milimeters
values = dataset.values[:,1]*1000 if useGps else dataset.values
print(values.shape)

# plot displacement values in milimeters
#plt.plot(values)
ndates = len(values)
#plt.xlabel('Time blocks of %i dates per location' % (ndates))
#plt.ylabel('Displacement (mm)')
#plt.title('Deformation MAY\'15 - DEC\'18')
#plt.show()

# select range of dates for years between MAY 2015 and DEC 2018
ndates = len(values)
useVal = False; earlySeason = False; plotFit = False; maxPredDays = 1
daysObs = 30#4          # set to 30 or 4 with or without interpolation, respectively
test_size = 365#60      # set to 365 or 60 with or without interpolation, respectively
nMonths = int(ndates/daysObs)+1   # number of total months in the data
print(nMonths)
months = 31          # number of total months to analyse

# specify columns (locations) to consider and to plot
groups = range(1)
# plot each location
#plt.figure()
#for group in groups:
#    plt.subplot(len(groups), 1, group+1)
#    plt.plot(values[group*ndates:(group+1)*ndates])
#    plt.title('Displacements for location '+str(groups.index(group)+1))
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

#plt.close()
# run experiments - learning and test for each month
np.random.seed(4)
# Save Scores
timeRMSE, timeVAR, timeMAE, timeMDAE, timeMRE, timeR2 = np.empty(shape=(months,test_size)), np.empty(shape=(months,test_size)), np.empty(shape=(months,test_size)), np.empty(shape=(months,test_size)), np.empty(shape=(months,test_size)), np.empty(shape=(months,test_size))
trainPredict, testPredict = [], []
if useVal: valPredict = []
multistep = True;
for month in range(0,months):    # maximum period of 3 years observed
    print("Learning from previous %i months ..." % (month+1))

    ### define set of parameters:
    # timesteps for past and future observations of LSTM
    # set observations in periods of less than 12 months. ensuring enough amount number of samples
    #if np.mod(month+1,12) == 0:
    #    look_back = test_size
    #    look_forward = test_size
    #else:
    look_back = daysObs*(np.mod((month),12)+1)
    look_forward = daysObs*(np.mod((month),12)+1)
    if useVal:
        while ndates+1-look_back-look_forward-test_size <= test_size:
            look_back -= 1
            look_forward -= 1

    # frame as multivar supervised learning for each location
    reframed = series_to_supervised(scaled, ndates, nfeatures, look_back, look_forward)
    # drop columns we don't want to predict
    print(reframed.head())
    print(reframed.shape)

    n_obs = look_back * nfeatures
    n_futObs = look_forward * nfeatures

    # split into train and test sets
    train = reframed.values[:-test_size+n_obs, :n_obs]
    test = reframed.values[-test_size+n_futObs-1:, -n_futObs:]
    print(train.shape,test.shape)

    n_samples = (month+1)*daysObs

    # Multiple Lag Timesteps
    ############################################################################################################################################
    ########################## https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ ############################
    ############################################################################################################################################
    # This will modify the LSTM' input and output shapes in [samples, observations, features times future observations]
    if multistep:
        test_X, test_y = test[:, :], test[:, 0]               # test set consists of future observations from the next samples; test sequence consists of the first future observations
        test_X = test_X.reshape((test_X.shape[0], n_futObs, nfeatures))
        if not useVal:
            val_X, val_y = [], []
            train_X = train[-n_samples:, :n_obs]                                # training set consists of the last samples and chosen number of observations
            train_X = train_X.reshape((train_X.shape[0], n_obs, nfeatures))
            train_y = train[-n_samples:, -nfeatures]                            # training sequence for testing consists of the last observation from the first training sample
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
            print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    if useVal:
        history = model.fit(train_X, train_y, epochs=10, batch_size=50, validation_data=(val_X, val_y), verbose=1, shuffle=False)
    else:
        history = model.fit(train_X, train_y, epochs=10, batch_size=50, verbose=1, shuffle=False)

    # show learning process
    #plt.close()
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()

    trainPredict.append(scalerCD.inverse_transform(model.predict(train_X)))
    testPredict.append(scalerCD.inverse_transform(model.predict(test_X)))
    trainY = scalerCD.inverse_transform([train_y])
    testY = scalerCD.inverse_transform([test_y])
    #plt.close()
    # plot prediction example over the first future observation
    #plt.title('Real and predicted Displacements for a location')
    #if useVal:
        # append prediction values over the first future observation only and invert back to the original scale
    #    valY = scalerCD.inverse_transform([val_y])
    #    plt.plot(valY[0], label='Real', color='blue')
    #    plt.plot(scalerCD.inverse_transform(model.predict(val_X)), label='Fit-'+str(len(val_y)), color='green')
    #else:
    #    plt.plot(testY[0], label='Real', color='blue')
    #    plt.plot(scalerCD.inverse_transform(model.predict(test_X)), label='Forecast-'+str(len(test_y)), color='green')
    #plt.xlabel('Day')
    #plt.ylabel('Cumulative Displacement (mm)')
    #plt.legend(bbox_to_anchor=(1, 1))
    #plt.show()
    #if useGps:
    #    plt.savefig('/home/cosc/vp17941/linux/results/satsense/seasonality/multistep/GPS_'+signals[s]+'/forecast/Hoober_'+str(month+1)+'prevmonths-'+signals[s]+'.png')
    #else:
    #    plt.savefig('/home/cosc/vp17941/linux/results/satsense/seasonality/multistep/'+filterlevels[fl]+'/forecast/Normanton_'+str(month+1)+'prevmonths-'+str(look_forward)+'days-Observed-Forecast_'+filterlevels[fl]+'.png', bbox_inches='tight')

    # Calculate regression scores for all metrics and future observations
    ## initialise errors with nan for the whole season to forecast
    timeRMSE[month], timeVAR[month], timeMAE[month], timeMDAE[month], timeMRE[month], timeR2[month] = np.full(test_size, np.nan), np.full(test_size, np.nan), np.full(test_size, np.nan), np.full(test_size, np.nan), np.full(test_size, np.nan),  np.full(test_size, np.nan)
    for i in range(n_futObs):                           # test samples's future observations
        test_y = test_X[:, (i+feat)*nfeatures, 0]       # consider locations' displacements only
        testY = scalerCD.inverse_transform([test_y])
        timeRMSE[month][i] = math.sqrt(mean_squared_error(testPredict[month][:, 0], testY[0]))
        #timeVAR[month][i] = explained_variance_score(testPredict[month][:, 0], testY[0])
        #timeMAE[month][i] = mean_absolute_error(testPredict[month][:, 0], testY[0])
        #timeMDAE[month][i] = median_absolute_error(testPredict[month][:, 0], testY[0])
        #timeMRE[month][i] = max_error(testPredict[month][:, 0], testY[0])
        #timeR2[month][i] = r2_score(testPredict[month][:, 0], testY[0])
        if np.mod(i,test.shape[0]/10) == 0: print('Calculating scores ... %.2f%% completed' % (i/test.shape[0]*nfeatures*100))
    print('100%% done!')
    #np.argmin(timeRMSE),np.nanmin(timeRMSE),np.nanmax(timeRMSE),np.argmax(timeRMSE)

#print(timeRMSE)
#print(timeVAR)
#print(timeMAE)
#print(timeMDAE)
#print(timeMRE)
#print(timeR2)

visualiseForecast(plotFit, earlySeason, test_size, values, ndates, months, daysObs, trainPredict, testPredict, timeRMSE)
