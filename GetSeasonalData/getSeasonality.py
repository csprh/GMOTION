#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas, math, pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.io as sio

mat_contents = sio.loadmat('interpLocationNorm.mat')
interpLocation = mat_contents['interpLocation']
outcdTSmooth = interpLocation['outcdTSmooth']
cd = outcdTSmooth[0,0]
sh = np.shape(cd)
leng = sh[0]
sIndex = np.zeros(leng)
tIndex = np.zeros(leng)

for ii in range(leng):
    thisSeries = cd[ii,:]
    print(ii)
    result = seasonal_decompose(thisSeries, model='additive',freq=61, extrapolate_trend = 61)
    #print(result.trend)

    #print(result.seasonal)
    #print(result.resid)
    varR = np.var(result.resid)
    varRT = np.var(result.resid+result.trend)
    varSR = np.var(result.resid+result.seasonal)

    sIndex1 = 1.0-(varR/(varSR))
    tIndex1 = 1.0-(varR/(varRT))
    sIndex[ii] = np.maximum(0.0,sIndex1)
    tIndex[ii] = np.maximum(0.0,tIndex1)


    #plt.close()
    #plt.plot(result.trend, label='trend', color='blue')
    #plt.plot(result.seasonal, label='seasonal', color='red')
    #plt.plot(result.resid, label='resid', color='green')
    #plt.plot(result.resid+result.seasonal, label='observed', color='pink')

    #plt.xlabel('Day')                          # use for the averaged CDs
    #plt.ylabel('Cumulative Displacement')
    #plt.legend(bbox_to_anchor=(0.9, 1))
    #plt.show()
mat_contents['arrayS'] = sIndex
mat_contents['arrayT'] = tIndex

sio.savemat('interpLocationNorm2.mat', mat_contents)

