#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import pandas, math
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
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

# plot displacement values in milimeters
plt.plot(values)
ndates = len(values)
plt.xlabel('Time blocks of %i dates per location' % (ndates))
plt.ylabel('Displacement (mm)')
plt.title('Deformation MAY\'15 - DEC\'18')
plt.show()

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
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, group+1)
    plt.plot(values[group*ndates:(group+1)*ndates])
    plt.title('Displacements for location '+str(groups.index(group)+1))
plt.show()

# split data into groups of locations given the dates and total size of the considered dataset
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

plt.close()
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
    plt.close()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # make predictions over the first future observations only and invert back to the original scale
    trainPredict = scalerCD.inverse_transform(model.predict(train_X))
    testPredict = scalerCD.inverse_transform(model.predict(test_X))
    trainY = scalerCD.inverse_transform([train_y])    
    testY = scalerCD.inverse_transform([test_y])
    
    # plot prediction
    plt.close()
    plt.title('Real and predicted Displacements for a location')
    plt.plot(testY[0], label='Real')
    #plt.plot(trainPredict, label='Train predicted')
    plt.plot(testPredict, label='Predicted')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Displacement (mm)')
    plt.legend()
    plt.show()
    # save figure (change the path!)
    #plt.savefig('/home/cosc/vp17941/linux/results/satsense/seasonality/multistep/GPS_'+signals[s]+'/preds_Hoober_'+str(month+1)+'prevmonths-'+signals[s]+'.png')    
       
    # Calculate regression scores for all metrics and future observations
    n_futObs = nfeatures*look_forward
    testRMSE = np.zeros(shape=(int(n_futObs/nf)))
    testVAR = np.zeros(shape=(int(n_futObs/nf)))
    testMAE = np.zeros(shape=(int(n_futObs/nf)))
    testMDAE = np.zeros(shape=(int(n_futObs/nf)))
    testMRE = np.zeros(shape=(int(n_futObs/nf)))
    testR2 = np.zeros(shape=(int(n_futObs/nf)))
    test_X, test_y = test[:, :n_obs], test[:, -nfeatures*look_forward]
    test_X = test_X.reshape(test_X.shape[0], look_back, nfeatures)
    testPredict = model.predict(test_X)
    testPredict = scalerCD.inverse_transform(testPredict)
    for i in range(len(testRMSE)):          # test all future observations 
        test_y = test[:, -n_futObs+i*nf]    # only predict displacements
        testY = scalerCD.inverse_transform([test_y])
        testRMSE[i] = math.sqrt(mean_squared_error(testPredict[:, 0], testY[0]))
        testVAR[i] = explained_variance_score(testPredict[:, 0], testY[0])
        testMAE[i] = mean_absolute_error(testPredict[:, 0], testY[0])
        testMDAE[i] = median_absolute_error(testPredict[:, 0], testY[0])
        testMRE[i] = max_error(testPredict[:, 0], testY[0])
        testR2[i] = r2_score(testPredict[:, 0], testY[0])
        if np.mod(i,len(testRMSE)/10) == 0: print('Calculating scores ... %.2f%% completed' % (i/n_futObs*nf*100))
    print('100%% done!')
    np.argmin(testRMSE),np.min(testRMSE),np.max(testRMSE),np.argmax(testRMSE)
    
    # For each month, calculate mean scores over all future observations 
    timeRMSE[month] = np.mean(testRMSE)
    timeVAR[month] = np.mean(testVAR)
    timeMAE[month] = np.mean(testMAE)
    timeMDAE[month] = np.mean(testMDAE)
    timeMRE[month] = np.mean(testMRE)
    timeR2[month] = np.mean(testR2)
    
print(timeRMSE, np.max(timeRMSE), np.min(timeRMSE))
print(timeVAR, np.max(timeVAR), np.min(timeVAR))
print(timeMAE, np.max(timeMAE), np.min(timeMAE))
print(timeMDAE, np.max(timeMDAE), np.min(timeMDAE))
print(timeMRE, np.max(timeMRE), np.min(timeMRE))
print(timeR2, np.max(timeR2), np.min(timeR2))