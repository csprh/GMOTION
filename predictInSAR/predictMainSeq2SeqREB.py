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
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras import backend as be
import numpy as np
import math
from utils import calcErr,plotPredictions,normbygroup,getMethodPreds
import scipy.io as sio
from series_to_supervised import series_to_supervised
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from datetime import date
from keras.callbacks import ModelCheckpoint
from scipy import optimize
from reset_keras import reset_keras
import random



# train the model
def getModel(x1,x2,y1):
	# prepare data

    model = Sequential()
    model.add(LSTM(200,  input_shape=(x1, x2)))
    model.add(RepeatVector(y1))
    model.add(LSTM(200,  return_sequences=True))
    model.add(TimeDistributed(Dense(100)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    return model


def plotPredictions(seq, s, n, yhat, thisColor, plotSignal):
    # plot forecasting

    endValue = seq[s-1]
    yhat = np.concatenate([np.array([endValue]),yhat])
    if plotSignal==1:
        plt.plot(np.arange(1,len(seq)+1), seq, label='Real Sequence', color="black")
    plt.plot(np.arange(s,s+len(yhat)), yhat, label='Forecast-'+n, color=thisColor)
    plt.ylabel('Cumulative Displacement')


def trainModel(train_y, train_X, epochsIn, earlyStopping):
    model = getModel(train_X.shape[1], train_X.shape[2], train_y.shape[1])
    model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=True)
    return model

def predInv(model, test_X, scaler):
    y_hat = model.predict(test_X)
    y_hat = scaler.inverse_transform(y_hat[:,:,0])[0,:]
    return y_hat

def getLSTMPred(train_y, train_X, test_X, scaler, epochsIn, earlyStopping):
    model = trainModel(train_y, train_X, epochsIn, earlyStopping)
    y_hat = predInv(model, test_X, scaler)
    return y_hat, model

def calcErr(yhat, inv_y):
    rmse = math.sqrt(mean_squared_error(yhat, inv_y))
    return rmse

def genTrain(scaledCD, look_back, look_forward, sampleBound):

    train_X = []
    train_y = []
    nSamples = scaledCD.shape[0]

    for i in range(0,nSamples):
        scaled = scaledCD[i,:]
        train=scaled[:-sampleBound]

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
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    return train_y, train_X

matlab_datenum = [736095,736329,736569,736809,737049, 737289]
dates = []; offset_days = 366
for d in matlab_datenum:
    dates.append(date.fromordinal(int(d-offset_days)).strftime("%d %B %Y"))
xInds = [1, 40, 80, 120, 160, 200]

mat_contents = sio.loadmat('interpLocationNorm.mat')
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

scaler = MinMaxScaler(feature_range=(0, 1))

scaledCD = cdTSmooth.reshape((cdTSmooth.shape[0]*cdTSmooth.shape[1],1))
scaledCD = np.array(scaler.fit_transform(scaledCD))[:,0]
scaledCD = scaledCD.reshape(cdTSmooth.shape[0],cdTSmooth.shape[1])

sampleTime = 6          # distance in days between samples
yearInSamples = int(365.25/sampleTime)
nfeatures = 1
predInDays = 265        # 9 months

epochs = 2000

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

rmseLSTM1Array = np.array([])

sh = list(range(0,cdTSmooth.shape[0]))
random.shuffle(sh)
sampleBound = 44

predXX = (44, 88, 132)
predYY = (11, 22, 44)

for XX in range(0,3):
 for YY in range(0,3):
  predInSamplesX = predXX[XX]
  predInSamplesY = predYY[YY]

  for ii in range(0,310):
    chooseSeq = theseInds[-(ii+1)]
    #chooseSeq = sh[ii]

    values = cdTSmooth[chooseSeq, :]
    scaled = scaledCD[chooseSeq, :]
    ndates = len(values)

    if sampleBound == predInSamplesY:
       test_y = values[-sampleBound:]
    else:
       test_y = values[-sampleBound:-sampleBound+predInSamplesY]
    test_X  = scaled[(-predInSamplesX-sampleBound): -sampleBound]
    test_X = test_X.reshape((1, test_X.shape[0],  nfeatures))

    # define training period of n past observations

    singleTrain = scaled
    singleTrain = singleTrain[..., np.newaxis]
    singleTrain = singleTrain.transpose()

    train_y1, train_X1  = genTrain(singleTrain,predInSamplesX, predInSamplesY,  sampleBound)

    y_hatLSTM1, model =  getLSTMPred(train_y1, train_X1,  test_X, scaler, epochs,0)

    rmseLSTM1  = calcErr(y_hatLSTM1, test_y)
    s = ndates - sampleBound - 1

    plt.close()
    thisfig = plt.figure(figsize=(12,8))
    plotPredictions(values, s, "LSTM1: RMSE = " + str(rmseLSTM1), y_hatLSTM1, "green", 1)
    plt.xticks(xInds, dates, rotation=30)

    if ii == 0:
        rmseLSTM1Array  = rmseLSTM1
    else:
        rmseLSTM1Array  = np.vstack((rmseLSTM1Array, rmseLSTM1))

    np.save('REB_MOD1_X'+str(predInSamplesX)+'_Y'+str(predInSamplesY)+'.npy', rmseLSTM1Array)

    plt.legend(loc='best')

    thisfig.savefig('REB_MOD1_X'+str(predInSamplesX)+'_Y'+str(predInSamplesY)+".pdf", bbox_inches='tight')

    plt.close(); print('\n')

    tf.compat.v1.keras.backend.clear_session(); reset_keras()
    #be.clear_session(); reset_keras()
    print('100%% done of position '+str(chooseSeq))




