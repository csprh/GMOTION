#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:11:06 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import explained_variance_score
#from sklearn.metrics import max_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import median_absolute_error
#from sklearn.metrics import r2_score

def plotTrainWindows(v, n_obs, t=4, nplots=5):
    plt.close()
    plt.figure(figsize=(6,8))
    plt.title('Training windows at different timesteps')
    for i in range(nplots):
        ax = plt.subplot(nplots, 1, i+1)
        plt.title('t'+str(i*t-n_obs), y=0.7, loc='right')
        x = np.arange(i*t+1, n_obs+i*t+1)
        plt.plot(x, v[i*t:n_obs+i*t])
        plt.xticks(x)
        for label in ax.get_xaxis().get_ticklabels()[::2]:
            label.set_visible(False)
    plt.xlabel('Datapoints over time')    
    plt.ylabel('Cumulative Displacements (mm)')
    ax.yaxis.set_label_coords(-0.1,3.02)
    plt.show()
    
def calcErr(yhat, inv_y, scaler):
    inv_yhat = scaler.inverse_transform(yhat)[0,:]
    # invert scaling for actual
    #inv_y = scaler.inverse_transform(test_y.reshape(1,len(test_y)))[0,:]
    
    # Calculate regression scores for all metrics and future observations
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    #var.append(explained_variance_score(inv_y, inv_yhat))
    #mae.append(mean_absolute_error(inv_y, inv_yhat))
    #mdae.append(median_absolute_error(inv_y, inv_yhat))
    #mre.append(max_error(inv_y, inv_yhat))
    #r2.append(r2_score(inv_y, inv_yhat))
    return rmse, inv_yhat, inv_y
    
def plotPredictions(seq, s, n, yhat, inv_y, leg):
    # plot forecasting
    if leg == '':
        plt.close()
        plt.figure(figsize=(12,8))
        plt.plot(np.arange(1,len(seq)+1), seq, label='Real Sequence', color='blue')
        plt.plot(np.arange(s,s+len(yhat)), yhat, label='Forecast-'+n, color='green')
    else:
        plt.plot(np.arange(s,s+len(yhat)), yhat, label='Forecast-'+leg)
    plt.xlabel('Day')                          # use for the averaged CDs
    plt.ylabel('Cumulative Displacement')
    plt.legend(loc='best')
    #plt.legend(bbox_to_anchor=(0.9, 1))
    
    if leg == '': 
        plt.show()
        plt.close()
        plt.plot(np.arange(1,len(inv_y)+1), inv_y, label='Real Sequence')
        plt.plot(np.arange(1,len(yhat)+1), yhat, label='Forecast-'+n)
        plt.legend(loc='best')
        plt.show()

def normbygroup(dataset, ndates, values, nfeatures, useGps):
    # split data into groups of locations given the dates and total size of the considered dataset
    #groups = range(0, int(len(dataset)/ndates))
    #values = np.delete(values, range(len(groups)*ndates,len(values)), 0)      # delete end locations
    values = np.transpose(np.array(values, ndmin=2)) if useGps else np.array(values, ndmin=2) 
    print(values.shape)
    # ensure all data is float
    values = values.astype('float32')
    
    # normalize Cumulative Displacements by location group
    scaled = np.zeros(shape=(values.shape[0], values.shape[1], nfeatures))
    scaler = MinMaxScaler(feature_range=(0, 1))
    groups = range(0, values.shape[1])
    for group in groups:
        #scaled[group*ndates:(group+1)*ndates,0] = np.transpose(np.array(scaler.fit_transform(values[group*ndates:(group+1)*ndates]), ndmin=2))
        scaled[:,group,0] = np.array(scaler.fit_transform(values[:,group].reshape(values.shape[0],nfeatures)))[:,0]
    #scaled = np.transpose(scaled)
    return values, scaled, scaler

#from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import statsmodels.api as sm
import pandas
#from scipy import optimize

def getMethodPreds(train, n, sinusParams, sarimaParams):
    ###############################################################################
    #https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/# 
    ###############################################################################
    
    testPredict, labels, nMethods = [], [], 0
    
    # Naive approach
    y_hat = np.repeat(train[-1], n)
    testPredict.append(y_hat)
    nMethods += 1; labels.append('Naive')
    
    # Simple Average
    y_hat = np.repeat(train.mean(), n)
    testPredict.append(y_hat)
    nMethods += 1; labels.append('Simple Average')
    
    # Moving Average
    y_hat = np.repeat(pandas.DataFrame(train).rolling(int(n/2)).mean().iloc[-1][0], n)
    testPredict.append(y_hat)
    nMethods += 1; labels.append('Moving Average')
    
    # Simple Exponential Smoothing
    #fit1 = SimpleExpSmoothing(train).fit(smoothing_level=0.6,optimized=False)
    #y_hat = fit1.forecast(n)
    #testPredict.append(y_hat)
    #nMethods += 1; labels.append('Simple Exponential Smoothing')
    
    # Holt's Linear Trend 
    #sm.tsa.seasonal_decompose(train, freq=4).plot()
    #result = sm.tsa.stattools.adfuller(train[:,0])
    #plt.show()
    #fit1 = Holt(train).fit(smoothing_level = 0.55, smoothing_slope = 0.5)
    #y_hat = fit1.forecast(n)
    #testPredict.append(y_hat)
    #nMethods += 1; labels.append('Holts Linear')
    
    # Curve sinusoid
    def sinusoid_curve(x, a, b, c):
        return a * np.sin(b * x) + c
    y_hat = sinusoid_curve(np.arange(0,len(train)), *sinusParams)
    #params, params_covariance = optimize.curve_fit(sinusoid_curve, np.arange(0,len(train)), y_hat)
    #plt.plot(trainY, label='Train')
    #y_hat = sinusoid_curve(np.arange(0,len(train)), *params)
    #plt.close(); plt.plot(y_hat); plt.show()
    #plt.plot(scalerCD.inverse_transform(y_hat.reshape(1,len(y_hat)))[0,:], label='Fitted function')
    testPredict.append(y_hat[:n])
    nMethods += 1; labels.append('Sinusoid fitting')
            
    # Holt-Winters add
    #fit1 = ExponentialSmoothing(train, seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)
    #y_hat = fit1.forecast(n)
    #testPredict.append(y_hat)
    #nMethods += 1; labels.append('Holts Winter -add-'); 
    
    # Holt-Winters mul
    #fit1 = ExponentialSmoothing(train, seasonal_periods=3, trend='add', seasonal='mul').fit(use_boxcox=True)
    #y_hat = fit1.forecast(n)
    #testPredict.append(y_hat)
    #nMethods += 1; labels.append('Holts Winter -mul-')
    
    ## SARIMA: 
    # - The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters. 
    # d must be an integer indicating the integration order of the process, while p and q may either be an integers 
    # indicating the AR and MA orders (so that all lags up to those orders are included) or else iterables giving 
    # specific AR and / or MA lags to include. Default is an AR(1) model: (1,0,0).
    # - The (P,D,Q,s) order of the seasonal component of the model for the AR parameters, differences, MA parameters, 
    # and periodicity. d must be an integer indicating the integration order of the process, while p and q may either 
    # be an integers indicating the AR and MA orders (so that all lags up to those orders are included) or else iterables 
    # giving specific AR and / or MA lags to include. s is an integer giving the periodicity (number of periods in season), 
    # often it is 4 for quarterly data or 12 for monthly data. Default is no seasonal effect.
    fit1 = sm.tsa.statespace.SARIMAX(train, order=(sarimaParams[:3]), seasonal_order=(sarimaParams[3:])).fit()
    y_hat = fit1.predict(start=len(train), end=len(train)+n-1, dynamic=True)
    testPredict.append(y_hat)
    nMethods += 1; labels.append('SARIMA')
    
    return testPredict, nMethods, labels