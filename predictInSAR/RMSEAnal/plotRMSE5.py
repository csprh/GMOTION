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




thisPos = 0
numberProcessed = 310


def showPlot(thisPos, plotName, plotTitle):


    l11 = np.load('LSTM1_Hollin1.npy')
    l12 = np.load('LSTM1_Hollin2.npy')
    l13 = np.load('LSTM1_Hollin3.npy')
    l21 = np.load('Sarima_Hollin1.npy')
    l22 = np.load('Sarima_Hollin2.npy')
    l23 = np.load('Sarima_Hollin3.npy')
    l31 = np.load('Sinu_Hollin1.npy')
    l32 = np.load('Sinu_Hollin2.npy')
    l33 = np.load('Sinu_Hollin3.npy')

    l5In = np.concatenate(([l11, l12, l13]), axis=0)
    l9In = np.concatenate(([l21, l22, l23]), axis=0)
    l10In = np.concatenate(([l31, l32, l33]), axis=0)
    for ii in range(0,9):
        l5_0 = l5In[:,ii]
        l9_0 = l9In[:,ii]
        l10_0 = l10In[:,ii]

        mInd = ((np.round(1+ii*np.ones([len(l5_0),]))))
        mInd =  mInd.astype(int)
        thisdataset = pd.DataFrame({'LSTM1': l5_0,  'Sarima': l9_0, 'Sinu': l10_0, 'Months Predicted': mInd} )
        if ii == 0:
            dataset = thisdataset
        else:
            dataset = pd.concat([dataset,thisdataset])

    thisfig = plt.figure(figsize=(12,8))
    dfNew = pd.melt(dataset, id_vars=['Months Predicted'], value_vars=['LSTM1', 'Sarima', 'Sinu'])
    sns.set(style="whitegrid")
    dfNew.rename({'variable': 'Prediction method'}, axis=1, inplace=True)

    ax = sns.boxplot(data=dfNew, x='Prediction method', y='value', hue = 'Months Predicted', showfliers=False)
    plt.ylabel('RMSE (Distribution in Quartiles: For 2000 Signals)')
    plt.title(plotTitle)
    thisfig.savefig(plotName, bbox_inches='tight')

    plt.show()

showPlot(0,'GroupPlotHollin.pdf','Prediction over 9 Months')


