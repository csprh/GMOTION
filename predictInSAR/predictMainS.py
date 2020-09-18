#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as be
import numpy as np
import math
from utils import calcErr,plotPredictions,normbygroup,getMethodPreds
import scipy.io as sio
from series_to_supervised import series_to_supervised
from pyramid.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import date
from scipy import optimize

def sinFunc(x, a, b, c):
    thisFreq = int(365.25/6)
    return a*x + b * np.sin((2*math.pi*x/thisFreq) + c)

def getSarimaPred(train, yearInSamples, predSamples):

   thissarima =  auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p= 3, max_q= 3, max_d = 3,
                      m=yearInSamples,
                      start_P=0,
                      max_D=2, max_Q= 2, max_P = 2,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True)

   y_hat, confint = thissarima.predict(n_periods=predSamples, return_conf_int=True)
   return y_hat

def getFittedSinPred(y_data, yearInSamples, predSamples):

    x_data  = np.array(range(0,len(y_data)))
    x_pred =  np.array(range(len(y_data),len(y_data)+predSamples))
    params, params_covariance = optimize.curve_fit(sinFunc, x_data, y_data,
                                               p0=[0, 5, 0])
    y_hat = sinFunc(x_pred, params[0], params[1], params[2])
    return y_hat

def getModel(x1,x2,y1):

   model = Sequential()

   # 1D_1D architecture -- values empirically tested in several signals
   model.add(LSTM(256, input_shape=(x1, x2), return_sequences=True))
   model.add(Dropout(0.1))      # 0.605 (0.25,0.25,0.25); 0.899 (0.25,0.1,0.25); 1.369 (0.5,0.1,0.1); # 0.917 (0.25,0.5,0.25); 0.647 (0.1,0.25,0.25)
   model.add(LSTM(128, return_sequences=False))
   model.add(Dropout(0.2))      # 0.5 (0.1,0.2,0.25); 0.507 (0.1,0.1,0.25); 0.853  (0.1,0.1,0.20); 0.642 (0.1,0.15,0.25); 0.901 (0.1,0.15,0.20);
   model.add(Dense(128))
   model.add(Dropout(0.25))     # 0.877 (0.25,0.25,0.1); 1.2 (0.25,0.1,0.1); 1.071 (0.1,0.1,0.1); 0.655 (0.1,0.25,0.25); 0.855 (0.1,0.25,0.25); 0.576 (0.15,0.2,0.25)
   model.add(Dense(y1))
   model.compile(loss='mse', optimizer='adam')

   return model

def plotPredictions(seq, s, n, yhat, thisColor, plotSignal):
    # plot forecasting

    endValue = seq[s-1]
    yhat = np.concatenate([np.array([endValue]),yhat])
    if plotSignal==1:
        plt.plot(np.arange(1,len(seq)+1), seq, label='Real Sequence', color="black")
    plt.plot(np.arange(s,s+len(yhat)), yhat, label='Forecast-'+n, color=thisColor)

    plt.xlabel('Samples (every 6 days)')                          # use for the averaged CDs
    plt.ylabel('Cumulative Displacement')



def getLSTMPred(train_y, train_X, test_X, scaler, epochsIn):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    be.tensorflow_backend.set_session(tf.Session(config=config))

    model = getModel(train_X.shape[1], train_X.shape[2], train_y.shape[1])

    history = model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=False)

    y_hat = model.predict(test_X)
    y_hat = scaler.inverse_transform(y_hat)[0,:]
    return y_hat

def calcErr(yhat, inv_y):
    rmse = math.sqrt(mean_squared_error(yhat, inv_y))
    return rmse

def genTrain(scaledCD, predInSamples):

    train_X = []
    train_y = []
    look_back = predInSamples
    look_forward = predInSamples
    nSamples = scaledCD.shape[0]

    for i in range(0,nSamples):
        scaled = scaledCD[i,:]
        train=scaled[:-predInSamples]

        train=train.reshape(len(train),1)
        trainSS = series_to_supervised(train, 1, look_back, look_forward)

        this_train_y = trainSS.values[:, -look_forward:]
        this_train_X = trainSS.values[:, :look_back];
        if i == 0:
            train_y = this_train_y
            train_X = this_train_X
        else:
            train_y = np.concatenate((train_y,this_train_y), axis=0)
            train_X = np.concatenate((train_X,this_train_X), axis=0)
    train_X = train_X.reshape((train_X.shape[0], look_back, 1))
    #train_y = train_y.reshape((train_y.shape[0], look_back, 1))
    return train_y, train_X

matlab_datenum = [736095,736329,736569,736809,737049, 737289]
dates = []; offset_days = 366
for d in matlab_datenum:
    dates.append(date.fromordinal(int(d-offset_days)).strftime("%d %B %Y"))
xInds = [1, 40, 80, 120, 160, 200]

mat_contents = sio.loadmat('interpLocationNorm.mat')
interpLocationStruct = mat_contents['interpLocation']
cdTSmooth = interpLocationStruct['outcdTSmooth'][0,0]
arrayAC = interpLocationStruct['arrayAC'][0,0][0,:]
arrayS = interpLocationStruct['arrayS'][0,0][0,:]
arraySin = interpLocationStruct['arraySin'][0,0][0,:]


theseInds = np.argsort(arrayAC)
sh = np.shape(cdTSmooth)
nDates = sh[0]
nPoints = sh[1]
nPoints5p = int(nPoints/5)

scaler = MinMaxScaler(feature_range=(0, 1))

scaledCD = cdTSmooth.reshape((cdTSmooth.shape[0]*cdTSmooth.shape[1],1))
scaledCD = np.array(scaler.fit_transform(scaledCD))[:,0]
scaledCD = scaledCD.reshape(cdTSmooth.shape[0],cdTSmooth.shape[1])

sampleTime = 6          # distance in days between samples
yearInSamples = int(365.25/sampleTime)
nfeatures = 1
predInDays = 265        # 9 months
predInSamples = int(predInDays/sampleTime)
epochs = 100

for ii in range(0,6):
    chooseSeq = theseInds[-(ii+1)]

    values = cdTSmooth[chooseSeq, :]
    scaled = scaledCD[chooseSeq, :]
    ndates = len(values)

    test_y = values[-predInSamples:]
    test_X  = scaled[(-predInSamples*2): -predInSamples]
    test_X = test_X.reshape((1, test_X.shape[0],  nfeatures))

    # define training period of n past observations

    singleTrain = scaled
    singleTrain = singleTrain[..., np.newaxis]
    singleTrain = singleTrain.transpose()

    train_y1, train_X1  = genTrain(singleTrain,predInSamples)
    train_y6, train_X6  = genTrain(scaledCD[theseInds[-6:], :],predInSamples)
    train_y5p, train_X5p = genTrain(scaledCD[theseInds[-nPoints5p:], :],predInSamples)

    #y_hatLSTM1 =  getLSTMPred(train_y1, train_X1,  test_X, scaler, epochs)
    #y_hatLSTM6 =  getLSTMPred(train_y6, train_X6, test_X, scaler,epochs)
    #y_hatLSTM10 = getLSTMPred(train_y5p, train_X5p, test_X, scaler,epochs)
    y_hatSarima = getSarimaPred(values[:-predInSamples], yearInSamples, predInSamples*4)
    y_hatSin = getFittedSinPred(values[:-predInSamples], yearInSamples, predInSamples*4)
    #rmseLSTM1 = calcErr(y_hatLSTM1, test_y)
    #rmseLSTM6 = calcErr(y_hatLSTM6, test_y)
    #rmseLSTM5p = calcErr(y_hatLSTM5p, test_y)
    #rmseSarima = calcErr(y_hatSarima, test_y)

    s = ndates - predInSamples

    plt.close()
    thisfig = plt.figure(figsize=(6,4))
    #plotPredictions(values, s, "LSTM1: RMSE = " + str(rmseLSTM1), y_hatLSTM1, "green", 1)
    #plotPredictions(values, s, "LSTM2: RMSE = "+  str(rmseLSTM6), y_hatLSTM6, "blue", 0)
    #plotPredictions(values, s, "LSTM3: RMSE = "+  str(rmseLSTM5p), y_hatLSTM5p, "pink", 0)
    plotPredictions(values, s, "Sarima", y_hatSarima, "red", 1)
    plotPredictions(values, s, "Sinusoid", y_hatSin, "yellow", 0)
    plt.legend(loc='best')
    #plt.show()
    thisfig.savefig("Pred-S2-"+str(chooseSeq)+".pdf", bbox_inches='tight')
    plt.close(); print('\n')

    print('100%% done of position '+str(chooseSeq))
#3995 Fit ARIMA: order=(3, 0, 2) seasonal_order=(1, 1, 0, 60); AIC=200.721, BIC=222.886, Fit time=227.015 seconds
#3744 Fit ARIMA: order=(2, 0, 3) seasonal_order=(1, 1, 0, 60); AIC=216.476, BIC=238.642, Fit time=261.776 seconds
#2793 Fit ARIMA: order=(3, 0, 3) seasonal_order=(1, 1, 0, 60); AIC=220.868, BIC=245.804, Fit time=191.585 seconds
#3096 Fit ARIMA: order=(3, 0, 3) seasonal_order=(1, 1, 0, 60); AIC=137.052, BIC=161.988, Fit time=148.918 seconds
#3743 Fit ARIMA: order=(3, 0, 2) seasonal_order=(1, 1, 0, 60); AIC=222.435, BIC=244.600, Fit time=105.383 seconds
#9242 Fit ARIMA: order=(3, 0, 2) seasonal_order=(1, 1, 0, 60); AIC=171.782, BIC=193.947, Fit time=63.902 seconds
