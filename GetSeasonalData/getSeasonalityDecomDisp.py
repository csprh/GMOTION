#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas, math, pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.io as sio
from datetime import date

mat_contents = sio.loadmat('interpLocationNorm.mat')
interpLocation = mat_contents['interpLocation']
outcdTSmooth = interpLocation['outcdTSmooth']
cd = outcdTSmooth[0,0]
sh = np.shape(cd)
leng = sh[0]
sIndex = np.zeros(leng)
tIndex = np.zeros(leng)
matlab_datenum = [736095,736329,736569,736809,737049, 737289]
dates = []; offset_days = 366
for d in matlab_datenum:
    dates.append(date.fromordinal(int(d-offset_days)).strftime("%d %B %Y"))
xInds = [1, 40, 80, 120, 160, 200]

thisSeries = cd[3743,:]

result = seasonal_decompose(thisSeries, model='additive',freq=61, extrapolate_trend = 61)

matlab_datenum = [736095,736329,736569,736809,737049, 737289]
dates = []; offset_days = 366
for d in matlab_datenum:
    dates.append(date.fromordinal(int(d-offset_days)).strftime("%d %B %Y"))
xInds = [1, 40, 80, 120, 160, 200]


plt.close()
thisfig = plt.figure(figsize=(12,8))
plt.plot(result.trend, label='Trend (R)', color='blue')
plt.plot(result.seasonal, label='Seasonal (S)', color='red')
plt.plot(result.resid, label='Residual (L)', color='green')
plt.plot(thisSeries, label='observed', color='black')

plt.xlabel('Day')                          # use for the averaged CDs
plt.ylabel('Cumulative Displacement')
plt.legend(bbox_to_anchor=(0.9, 1))
plt.xticks(xInds, dates, rotation=30)
thisfig.savefig("Comps.pdf", bbox_inches='tight')

plt.show()

