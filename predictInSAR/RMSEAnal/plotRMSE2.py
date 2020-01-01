#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import math
import scipy.io as sio

import seaborn as sns
import pandas as pd

l1In = np.load('M-LSTM1.npy')
l2In = np.load('M-LSTM6.npy')
l3In = np.load('M-LSTM5p.npy')
l4In = np.load('M-LSTMM.npy')
l5In = np.load('M-Sarima.npy')
l6In = np.load('M-Sinu.npy')

thisPos = 0
l1 = l1In[:,thisPos]
l2 = l2In[:,thisPos]
l3 = l3In[:,thisPos]
l4 = l4In[:,thisPos]
l5 = l5In[:,thisPos]
l6 = l6In[:,thisPos]



rmses = [l1, l2, l3, l4, l5, l6]

dataset = pd.DataFrame({'LSTM1': l1, 'LSTM2': l2, 'LSTM3': l3, 'LSTM4': l4,   'Sarima': l5, 'Sinu': l6})

thisfig = plt.figure(figsize=(12,8))

sns.set(style="whitegrid")


ax = sns.boxplot(data=dataset, showfliers = False)
#ax = sns.swarmplot(data=dataset, color=".25")

ast = np.argsort(rmses, axis=0)
col1 = ast[0,:]
lenCol = len(ast)
lstm0 = np.mean(col1==0)
lstm1 = np.mean(col1==1)
lstm2 = np.mean(col1==2)
lstm3 = np.mean(col1==3)
lstm4 = np.mean(col1==4)

print ("lstm0= ",lstm0,"lstm1= ",lstm1,"lstm2= ",lstm2,"lstm3= ",lstm3,"lstm4= ",lstm4)
plt.ylabel('RMSE Distribution')
thisfig.savefig("RMSEs2.pdf", bbox_inches='tight')

plt.show()

#plt.plot(np.arange(1,len(seq)+1), seq, label='Real Sequence', color="black")
#plt.plot(np.arange(s,s+len(yhat)), yhat, label='Forecast-'+n, color=thisColor)

#plt.xlabel('Samples (every 6 days)')                          # use for the averaged CDs
#plt.ylabel('Cumulative Displacement')


