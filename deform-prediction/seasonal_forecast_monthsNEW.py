#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import pandas
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils import calcErr,plotPredictions,normbygroup,getMethodPreds

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
    #datafiles = ['Normanton_orig_may15-dec18_unfiltered.txt', 'Normanton_orig_may15-dec18_Filt.txt', 'Normanton_orig_may15-dec18_APS.txt', 'Normanton_orig_may15-dec18_TSmooth.txt'];
    ##datafiles = ['Leeds_interp1day_may15-dec18_unfiltered.txt', 'Leeds_interp1day_may15-dec18_Filt.txt', 'Leeds_interp1day_may15-dec18_APS.txt', 'Leeds_interp1day_may15-dec18_TSmooth.txt'];
    filterlevels = ['unfiltered', 'Filt', 'APS', 'TSmooth'];     fl = 3
    datafile = datafiles[fl]
    datafile = 'NTSmooth.txt'
    ## locations with highest seasonality: 4062,4058 for Normanton; 12994 for Leeds
    #dataset = pandas.read_csv(datafile, header=None, usecols=[4062], engine='python')
    dataset = pandas.read_csv(datafile, header=None, usecols=[4626,4058], engine='python')

#### show data
dataset, datafile

# set displacement values to milimeters
values = dataset.values[:,1]*1000 if useGps else dataset.values
values = values[0::6]
print(values.shape)

# plot displacement values in milimeters
plt.close()
plt.plot(values)
ndates = len(values)
plt.xlabel('Time blocks of %i dates per location' % (ndates))
plt.ylabel('Displacement (mm)')
plt.title('Deformation MAY\'15 - DEC\'18')
#plt.show()

# select range of dates for years between MAY 2015 and DEC 2018
ndates = len(values)
daysObs = 4#30          # set to 30 or 4 with or without interpolation, respectively
test_size = 60#365      # set to 365 or 60 with or without interpolation, respectively
nMonths = int(ndates/daysObs)+1   # number of total months in the data
print("%i total months" % (nMonths))
months = 15          # number of total months to analyse
nMethods = 0

assert months*daysObs <= test_size

# specify columns (locations) to consider and to plot
groups = range(values.shape[1])
# plot each location
plt.close()
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, group+1)
    #plt.plot(values[group*ndates:(group+1)*ndates])
    plt.plot(values[:,group])
plt.show()
plt.close()

values, scaled, scalerCD = normbygroup(dataset, ndates, values, 1, useGps)

print(scaled.shape)
np.min(scaled[:,0]),np.max(scaled[:,0])   # check data is normalised within range [0,1]

for pred in range(months,months+1):

    # define test set and test sequence of n future observations
    nsamples = daysObs*pred
    test = values[-test_size:, 0]
    test_y = test[:nsamples]

    errs, preds = [], []
    print("Forecast %i months ..." % (pred))

    # define training period of n past observations
    train=scaled[:-test_size, 0]

    sinusParams = [0.5, 0.054, 0.25]
    sarimaParams = [0,1,1, 0,0,int(nsamples/3),2]
    savepath = 'sarima-'+str(sarimaParams[0])+'-'+str(sarimaParams[1])+'-'
    savepath += str(sarimaParams[2])+'_'+str(sarimaParams[3])+'-'+str(sarimaParams[4])+'-'
    savepath += str(sarimaParams[5])+'-'+str(sarimaParams[6])+'/'

    testPredict, nMethods, labels = getMethodPreds(train, nsamples, sinusParams, sarimaParams)

    # Retrieve real train and test sequences for plotting
    train_y = train[:,0]

    # Calculate scores and plot
    rmse = np.zeros(shape=(nMethods,1))
    plt.close(); print('\n')
    plt.figure(figsize=(12,8))
    if test_size == nsamples:
        plt.plot(np.arange(1,len(values)+1), values, label='Real Sequence', color='blue')
    else:
        plt.plot(np.arange(1,len(values[:-test_size+nsamples])+1), values[:-test_size+nsamples], label='Real Sequence', color='blue')
    for i in range(nMethods):
        rmse[i,0], inv_yhat, inv_y = calcErr(testPredict[i].reshape(1, nsamples), test_y, scalerCD)         # calculate errors for every method
        plotPredictions(values[:,0], ndates-test_size, str(pred), inv_yhat, inv_y, labels[i])
        print('%s Approach, RMSE: %.8f' % (labels[i],rmse[i]))
        testPredict[i] = inv_yhat
    plt.show(); plt.close()
    errs.append(rmse[-1,0]), preds.append(testPredict[-1])

    # SAVE errors and predictions
    pickle.dump(errs, open(savepath+'errSARIMA'+str(pred)+'.pkl', 'wb'))
    pickle.dump(preds, open(savepath+'predSARIMA'+str(pred)+'.pkl', 'wb'))

print('100%% done!')



####################################################################################
# tail code, just for reading stored results and generate different visualisations #
####################################################################################




from plotConfidentInt import plotConfidentInt

dirs = ['sarima-0-1-1_0-0-20-2']
#,'sarima-0-1-2_0-0-20-2','sarima-8-0-8_0-0-15-2','sarima-8-0-8_0-0-20-2'

# Load saved data
months = 15; e,p = [],[]; pred=15
for d in dirs:
    e.append(pickle.load(open(d+'/errSARIMA'+str(pred)+'.pkl', 'rb')))
    p.append(pickle.load(open(d+'/predSARIMA'+str(pred)+'.pkl', 'rb')))
e
# Store new results for a specific forecasting period
pred = 9; nsamples = daysObs*pred
for pp in p:
    for ppp in pp:
        yhat = scalerCD.fit_transform(ppp[:nsamples].reshape(nsamples,1))
        err,inv_yhat,_ = calcErr(yhat.reshape(1,nsamples), test_y[:nsamples], scalerCD)
        pickle.dump([err], open(d+'/errSARIMA'+str(pred)+'.pkl', 'wb'))
        pickle.dump([inv_yhat], open(d+'/predSARIMA'+str(pred)+'.pkl', 'wb'))

# Load new data
e,p = [],[]
for d in dirs:
    e.append(pickle.load(open(d+'/errSARIMA'+str(pred)+'.pkl', 'rb')))
    p.append(pickle.load(open(d+'/predSARIMA'+str(pred)+'.pkl', 'rb')))

# Plot
i = [4626]
tr =  dataset[i].values[:-test_size].reshape(1,ndates-test_size)[0,:]
inv_y = inv_y[:nsamples]
meanOnly = False
plotConfidentInt([], p, inv_y, tr, [], daysObs, pred, [], 'all', meanOnly, dirs)
s = 'preds'+str(pred)+'_all_SIG1.png' if len(p)==1 else '/preds1-'+str(pred)+'_all_SIG1.png'
#plt.show()
plt.savefig(s)
plotConfidentInt([], p, inv_y, tr, [], daysObs, pred, [], 'superzoom', meanOnly, dirs)
s = 'preds'+str(pred)+'_zoom_SIG1.png' if len(p)==1 else '/preds1-'+str(pred)+'_zoom_SIG1.png'
#plt.show()
plt.savefig(s)
if len(p) > 1:
    plotConfidentInt(e, [], inv_y, [], [], daysObs, pred, [], 'superzoom', meanOnly, '')      # plot errors or no real seq.
elif len(p) == 1:
    plt.close()
    plt.figure(figsize=(12,8))
    plt.bar(np.arange(len(e)), np.array(e)[:,0], 0.5)
    plt.ylabel('Root Mean Square Error')
    plt.title('Mean error and Standard deviation of forecasting '+str(pred)+' months')
    plt.xticks(np.arange(len(e)), dirs, rotation=30)
    #plt.show()
    plt.savefig('barErrs'+str(pred)+'_SIG1-'+str(len(e))+'.png')
