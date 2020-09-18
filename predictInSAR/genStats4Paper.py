#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import calcErr,plotPredictions,normbygroup,getMethodPreds
import scipy.io as sio
from series_to_supervised import series_to_supervised
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import date
from scipy import optimize
from sklearn.metrics import mean_squared_error

def plotPredictions(seq, s, n, yhat, thisColor, plotSignal):
    # plot forecasting

    endValue = seq[s-1]
    yhat = np.concatenate([np.array([endValue]),yhat])
    if plotSignal==1:
        plt.plot(np.arange(1,len(seq)+1), seq, label='Real Sequence', color="black")
    plt.plot(np.arange(s,s+len(yhat)), yhat, label='Forecast-'+n, color=thisColor)
    plt.ylabel('Cumulative Displacement')

def calcErr(yhat, inv_y):
    rmse = math.sqrt(mean_squared_error(yhat, inv_y))
    return rmse

def sinFunc(x, a, b):
    return a*x + b

def getFittedLin1Pred(y_data, yearInSamples, predSamples):

    x_data  = np.array(range(0,len(y_data)))
    x_pred =  np.array(range(len(y_data),len(y_data)+predSamples))
    #params, params_covariance = optimize.curve_fit(sinFunc, x_data, y_data,
    #                                           p0=[0, 5, 0])
    params, params_covariance = optimize.curve_fit(sinFunc, x_data, y_data, p0=[0, 0])
    y_hat = sinFunc(x_pred, params[0], params[1])
    return y_hat

def getFittedLin2Pred(y_data, yearInSamples, predSamples):

    endPoint = y_data[-1]
    y_hat = np.ones(predSamples) * endPoint
    return y_hat

matlab_datenum = [736095,736329,736569,736809,737049, 737289]
dates = []; offset_days = 366
for d in matlab_datenum:
    dates.append(date.fromordinal(int(d-offset_days)).strftime("%d %B %Y"))
xInds = [1, 40, 80, 120, 160, 200]

mat_contents = sio.loadmat('interpLocationNorm.mat')
mat_indices = sio.loadmat('outputList2.mat')
outputList2 = mat_indices['ouputList2']
interpLocationStruct = mat_contents['interpLocation']
cdTSmooth = interpLocationStruct['outcdTSmooth'][0,0]
arrayLat = interpLocationStruct['lat2'][0,0][:,0]
arrayLon = interpLocationStruct['lon2'][0,0][:,0]
arrayAC = interpLocationStruct['arrayAC'][0,0][0,:]
arrayS = interpLocationStruct['arrayS'][0,0][0,:]
arraySin = interpLocationStruct['arraySin'][0,0][0,:]


theseInds = np.argsort(arrayAC)
sh = np.shape(cdTSmooth)
nPoints = sh[0]
nPoints5p = int(nPoints/100)


sampleTime = 6          # distance in days between samples
yearInSamples = int(365.25/sampleTime)
nfeatures = 1
predInDays = 265        # 9 months
predInSamples = int(predInDays/sampleTime)
epochs = 2000
noMSamples = 8

rmseSinArray = np.array([])
seasonal = 1

for ii in range(0,2000):
    if seasonal == 1:
        chooseSeq = theseInds[-(ii+1)]
        PreString = 'Seas'
    else:
        chooseSeq = np.int(outputList2[ii])
        PreString = 'Rand'
    values = cdTSmooth[chooseSeq, :]
    ndates = len(values)

    test_y = values[-predInSamples:]


    y_hatLin1    = getFittedLin1Pred(values[:-predInSamples], yearInSamples, predInSamples)
    y_hatLin2    = getFittedLin2Pred(values[:-predInSamples], yearInSamples, predInSamples)

    thisVar = np.var(values[:-predInSamples])
    thisRange = np.max(values[:-predInSamples])-np.max(values[:-predInSamples])


    rmseLin1 = np.zeros(9)
    rmseLin2 = np.zeros(9)
    for ind in range(1,9):
        rmseLin1[ind-1]    = calcErr(y_hatLin1[0:ind*5], test_y[0:ind*5])
        rmseLin2[ind-1]    = calcErr(y_hatLin2[0:ind*5], test_y[0:ind*5])
    rmseLin1[8]    = calcErr(y_hatLin1,  test_y)
    rmseLin2[8]    = calcErr(y_hatLin2,  test_y)
    s = ndates - predInSamples

    #plt.close()
    #thisfig = plt.figure(figsize=(12,8))
    #plotPredictions(values, s, "Lin1: RMSE = " +str(rmseLin1[8]), y_hatLin1, "yellow", 1)
    #plotPredictions(values, s, "Lin2: RMSE = " +str(rmseLin2[8]), y_hatLin2, "red", 0)
    #plt.xticks(xInds, dates, rotation=30)

    if ii == 0:
        rmseLin1Array = rmseLin1
        rmseLin2Array = rmseLin2
        varArray = thisVar
        rangeArray = thisRange
    else:
        rmseLin1Array = np.vstack((rmseLin1Array, rmseLin1))
        rmseLin2Array = np.vstack((rmseLin2Array, rmseLin2))
        varArray = np.vstack((varArray, thisVar))
        rangeArray = np.vstack((rangeArray, thisRange))


    np.save(PreString + 'M-Lin1.npy', rmseLin1Array)
    np.save(PreString + 'M-Lin2.npy', rmseLin2Array)
    np.save(PreString + 'M-var.npy', varArray)
    np.save(PreString + 'M-range.npy', rangeArray)

    #plt.legend(loc='best')
    #plt.show()
    #thisfig.savefig("TestLin.pdf", bbox_inches='tight')
    #plt.close(); print('\n')
    print('100%% done of position '+str(chooseSeq))



