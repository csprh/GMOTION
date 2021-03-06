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


def showPlot(thisPos, plotName, plotTitle, normal):


    l5In = np.load('M-Seq2Seq-LSTM1_R.npy')
    l8In = np.load('M-Seq2Seq-LSTMM_R.npy')
    l9In = np.load('M-Seq2Seq-Sarima_R.npy')
    l10In = np.load('M-Seq2Seq-Sinu_R.npy')
    l11In = np.load('RandM-Lin1.npy')
    l12In = np.load('RandM-Lin2.npy')
    for ii in range(0,9):
        l5_0 = l5In[:,ii]
        l8_0 = l8In[:,ii]
        l9_0 = l9In[:,ii]
        l10_0 = l10In[:,ii]
        l11_0 = l11In[:,ii]
        l12_0 = l12In[:,ii]

        varIn = np.load('SeasM-var.npy')
        lvar = np.squeeze(varIn)
        #lvar = l12_0

        l8_0_n = np.divide(l8_0, lvar)
        l9_0_n = np.divide(l9_0, lvar)
        l10_0_n = np.divide(l10_0, lvar)
        l12_0_n = np.divide(l12_0, lvar)
        mInd = ((np.round(1+ii*np.ones([len(l5_0),]))))
        mInd =  mInd.astype(int)
        if normal == 1:
            thisdataset = pd.DataFrame({'Constant': l12_0, 'Seq2Seq4': l8_0,  'Sarima': l9_0, 'Sinu': l10_0, 'Months Predicted': mInd} )
        else:
            thisdataset = pd.DataFrame({'Constant': l12_0_n,  'Seq2Seq4': l8_0_n,  'Sarima': l9_0_n, 'Sinu': l10_0_n, 'Months Predicted': mInd} )

        if ii == 0:
            dataset = thisdataset
        else:
            dataset = pd.concat([dataset,thisdataset])

    thisfig = plt.figure(figsize=(9,6))
    dfNew = pd.melt(dataset, id_vars=['Months Predicted'], value_vars=['Constant', 'Sinu', 'Sarima', 'Seq2Seq4'])
    sns.set(style="whitegrid")
    dfNew.rename({'variable': 'Prediction method'}, axis=1, inplace=True)

    ax = sns.boxplot(data=dfNew, x='Prediction method', y='value', hue = 'Months Predicted', showfliers=False)
    #ax.legend(loc='upper left')
    if normal == 1:
        plt.ylabel('RMSE (Distribution in Quartiles: For 2000 Signals)')
    else:
        plt.ylabel('n1RMSE (Distribution in Quartiles: For 2000 Signals)')
    plt.title(plotTitle)

    thisfig.savefig(plotName, bbox_inches='tight')

    plt.show()

showPlot(0,'GroupPlotNormalVar.pdf','2000 Randomly Selected Time Series', 0 )
showPlot(0,'GroupPlotNotNormal.pdf','2000 Randomly Selected Time Series', 1 )


