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
from keras.callbacks import EarlyStopping
from datetime import date

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

def getModel(x1,x2,y1):

   model = Sequential()

   # 1D_1D architecture -- values empirically tested in several signals
   model.add(LSTM(256, input_shape=(x1, x2), return_sequences=True))
   model.add(Dropout(0.1))
   model.add(LSTM(128, return_sequences=False))
   model.add(Dropout(0.2))
   model.add(Dense(128))
   model.add(Dropout(0.25))
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


def trainModel(train_y, train_X, epochsIn, earlyStopping):

    model = getModel(train_X.shape[1], train_X.shape[2], train_y.shape[1])

    if earlyStopping == 1:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)
        # fit model
        history = model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=False, validation_split=0.3, callbacks=[es])
    else:
        history = model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=False)
    return model

def predInv(model, test_X, scaler):
    y_hat = model.predict(test_X)
    y_hat = scaler.inverse_transform(y_hat)[0,:]
    return y_hat

def getLSTMPred(train_y, train_X, test_X, scaler, epochsIn, earlyStopping):
    model = trainModel(train_y, train_X, epochsIn, earlyStopping)
    y_hat = predInv(model, test_X, scaler)
    return y_hat, model

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


mat_contents = sio.loadmat('interpLocationNorm.mat')
interpLocationStruct = mat_contents['interpLocation']
cdTSmooth = interpLocationStruct['outcdTSmooth'][0,0]
arrayAC = interpLocationStruct['arrayAC'][0,0][0,:]
arrayS = interpLocationStruct['arrayS'][0,0][0,:]
arraySin = interpLocationStruct['arraySin'][0,0][0,:]


theseInds = np.argsort(arrayAC)
sh = np.shape(cdTSmooth)
nPoints = sh[0]
nPoints5p = int(nPoints/20)

scaler = MinMaxScaler(feature_range=(0, 1))

scaledCD = cdTSmooth.reshape((cdTSmooth.shape[0]*cdTSmooth.shape[1],1))
scaledCD = np.array(scaler.fit_transform(scaledCD))[:,0]
scaledCD = scaledCD.reshape(cdTSmooth.shape[0],cdTSmooth.shape[1])

sampleTime = 6          # distance in days between samples
yearInSamples = int(365.25/sampleTime)
nfeatures = 1
predInDays = 265        # 9 months
predInSamples = int(predInDays/sampleTime)
epochs = 5

train_y6, train_X6  = genTrain(scaledCD[theseInds[-6:], :],predInSamples)
train_y5p, train_X5p = genTrain(scaledCD[theseInds[-nPoints5p:], :],predInSamples)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
be.tensorflow_backend.set_session(tf.Session(config=config))
model_y6 =  trainModel(train_y6, train_X6, epochs, 0)
model_y6.save_weights(filepath  = 'y6.h5')
model_y5p =  trainModel(train_y5p, train_X5p, epochs, 0)
model_y5p.save_weights(filepath  = 'y5p2.h5')

rmseLSTM1Array = np.array([])
rmseLSTM6Array = np.array([])
rmseLSTM5pArray = np.array([])
rmseSarimaArray = np.array([])

for ii in range(0,1000):
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

    y_hatLSTM1, model =  getLSTMPred(train_y1, train_X1,  test_X, scaler, epochs,0)

    model.load_weights('y6.h5')
    y_hatLSTM6 =  predInv(model, test_X, scaler)
    model.load_weights('y5p2.h5')
    y_hatLSTM5p = predInv(model, test_X, scaler)

    y_hatSarima = getSarimaPred(values[:-predInSamples], yearInSamples, predInSamples)
    rmseLSTM1 = calcErr(y_hatLSTM1, test_y)
    rmseLSTM6 = calcErr(y_hatLSTM6, test_y)
    rmseLSTM5p = calcErr(y_hatLSTM5p, test_y)
    rmseSarima = calcErr(y_hatSarima, test_y)

    s = ndates - predInSamples

    plt.close()
    thisfig = plt.figure(figsize=(12,8))
    plotPredictions(values, s, "LSTM1: RMSE = " + str(rmseLSTM1), y_hatLSTM1, "green", 1)
    plotPredictions(values, s, "LSTM2: RMSE = "+  str(rmseLSTM6), y_hatLSTM6, "blue", 0)
    plotPredictions(values, s, "LSTM3: RMSE = "+  str(rmseLSTM5p), y_hatLSTM5p, "pink", 0)
    plotPredictions(values, s, "Sarima: RMSE = " +str(rmseSarima), y_hatSarima, "red", 0)

    rmseLSTM1Array = np.append(rmseLSTM1Array, rmseLSTM1)
    rmseLSTM6Array = np.append(rmseLSTM6Array, rmseLSTM6)
    rmseLSTM5pArray = np.append(rmseLSTM5pArray, rmseLSTM5p)
    rmseSarimaArray = np.append(rmseSarimaArray, rmseSarima)

    np.save('LSTM1.npy', rmseLSTM1Array)
    np.save('LSTM6.npy', rmseLSTM6Array)
    np.save('LSTM5p.npy', rmseLSTM5pArray)
    np.save('Sarima.npy', rmseSarimaArray)

    plt.legend(loc='best')
    plt.xticks(np.arange(len(dates)), dates, rotation=30)
    #plt.show()
    thisfig.savefig("Pred-"+str(chooseSeq)+".pdf", bbox_inches='tight')
    plt.close(); print('\n')

    print('100%% done of position '+str(chooseSeq))



