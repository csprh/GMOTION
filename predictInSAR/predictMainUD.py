#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as be
from keras.models import load_model
import numpy as np
import math
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping


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


def getLSTMPred(train_y, train_X, test_X, scaler, epochsIn, LSTM, saveData, saveModel):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    be.tensorflow_backend.set_session(tf.Session(config=config))

    if LSTM == 1 :
        history = model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=False)
    if LSTM == 2 :
        if saveData == 1 :
            np.save('LSTM2_train_y.npy', train_y)
            np.save('LSTM2_train_X.npy', train_X)
            return 0
        else:
            train_y = np.load('LSTM2_train_y.npy')
            train_X = np.load('LSTM2_train_X.npy')
        if saveModel == 1 :
            # fit model
            history = model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=False)
            model.save("LSTM2.h5")
        else:
            model = load_model("LSTM2.h5")
    if LSTM == 3 :
        if saveData == 1 :
            np.save('LSTM3_train_y.npy', train_y)
            np.save('LSTM3_train_X.npy', train_X)
            return 0
        else:
            train_y = np.load('LSTM3_train_y.npy')
            train_X = np.load('LSTM3_train_X.npy')
        if saveModel == 1 :
            # fit model
            history = model.fit(train_X, train_y, epochs=epochsIn, batch_size=128, verbose=1, shuffle=False)
            model.save("LSTM3.h5")
        else:
            model = load_model("LSTM3.h5")
    y_hat = model.predict(test_X)
    y_hat = scaler.inverse_transform(y_hat)[0,:]
    return y_hat


def calcErr(yhat, inv_y):
    rmse = math.sqrt(mean_squared_error(yhat, inv_y))
    return rmse


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
epochs = 2000

saveData = 0
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

    if ii == 0 :
        saveModel = 1
    else :
        saveModel = 0


    y_hatLSTM6 =  getLSTMPred(0, 0, test_X, scaler,epochs,2, saveData,saveModel)
    y_hatLSTM5p = getLSTMPred(0, 0, test_X, scaler,epochs,3, saveData,saveModel)



# Sing to god
# Day is gone
# Du hast
# Bastille
# Merry christmas
# Passion
# God is a DJ
# Walk right in
# Electro tracks *2

