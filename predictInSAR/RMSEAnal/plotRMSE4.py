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


    l5In = np.load('M-Seq2Seq-LSTM1_R.npy')
    l8In = np.load('M-Seq2Seq-LSTMM_R.npy')
    l9In = np.load('M-Seq2Seq-Sarima_R.npy')
    l10In = np.load('M-Seq2Seq-Sinu_R.npy')

    for ii in range(0,9):
        l5_0 = l5In[:,ii]
        l8_0 = l8In[:,ii]
        l9_0 = l9In[:,ii]
        l10_0 = l10In[:,ii]

        mInd = ((np.round(1+ii*np.ones([len(l5_0),]))))
        mInd =  mInd.astype(int)
        thisdataset = pd.DataFrame({'LSTM5': l5_0,  'LSTM8': l8_0, 'Sarima': l9_0, 'Sinu': l10_0, 'Months Predicted': mInd} )
        if ii == 0:
            dataset = thisdataset
        else:
            dataset = pd.concat([dataset,thisdataset])

    thisfig = plt.figure(figsize=(12,8))
    dfNew = pd.melt(dataset, id_vars=['Months Predicted'], value_vars=['LSTM5', 'LSTM8', 'Sarima', 'Sinu'])
    sns.set(style="whitegrid")
    dfNew.rename({'variable': 'Prediction method'}, axis=1, inplace=True)

    ax = sns.boxplot(data=dfNew, x='Prediction method', y='value', hue = 'Months Predicted')
    plt.ylabel('RMSE (Distribution in Quartiles and Outliers: For 210 Signals)')
    plt.title(plotTitle)
    thisfig.savefig(plotName, bbox_inches='tight')

    plt.show()

showPlot(0,'GroupPlot2.pdf','Prediction over 9 Months')


