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

    l1In = np.load('M-LSTM1.npy')
    l2In = np.load('M-LSTM6.npy')
    l3In = np.load('M-LSTM5p.npy')
    l4In = np.load('M-LSTMM.npy')
    l5In = np.load('M-Seq2Seq-LSTM1.npy')
    l6In = np.load('M-Seq2Seq-LSTM6.npy')
    l7In = np.load('M-Seq2Seq-LSTM5p.npy')
    l8In = np.load('M-Seq2Seq-LSTMM.npy')
    l9In = np.load('M-Sarima.npy')
    l10In = np.load('M-Sinu.npy')
    l11In = np.load('SeasM-Lin1.npy')
    l12In = np.load('SeasM-Lin2.npy')

    rangeIn = np.load('SeasM-range.npy')
    varIn = np.load('SeasM-var.npy')

    l1 = l1In[0:numberProcessed,thisPos]
    l2 = l2In[0:numberProcessed,thisPos]
    l3 = l3In[0:numberProcessed,thisPos]
    l4 = l4In[0:numberProcessed,thisPos]
    l5 = l5In[0:numberProcessed,thisPos]
    l6 = l6In[0:numberProcessed,thisPos]
    l7 = l7In[0:numberProcessed,thisPos]
    l8 = l8In[0:numberProcessed,thisPos]
    l9 = l9In[0:numberProcessed,thisPos]
    l10 = l10In[0:numberProcessed,thisPos]
    l11 = l11In[0:numberProcessed,thisPos]
    l12 = l12In[0:numberProcessed,thisPos]

    lvar = np.squeeze(varIn[0:numberProcessed])
    lrange = np.squeeze(rangeIn[0:numberProcessed])
    rmses = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
    lvar = l12
    l1_n = np.divide(l1, lvar)
    l2_n = np.divide(l2, lvar)
    l3_n = np.divide(l3, lvar)
    l4_n = np.divide(l4, lvar)
    l5_n = np.divide(l5, lvar)
    l6_n = np.divide(l6, lvar)
    l7_n = np.divide(l7, lvar)
    l8_n = np.divide(l8, lvar)
    l9_n = np.divide(l9, lvar)
    l10_n = np.divide(l10, lvar)
    l12_n = np.divide(l12, lvar)
    if normal == 1:
        dataset = pd.DataFrame({'Constant': l12, 'LSTM1': l1, 'LSTM2': l2, 'LSTM3': l3, 'LSTM4': l4, 'Seq2Seq1': l5, 'Seq2Seq2': l6, 'Seq2Seq3': l7, 'Seq2Seq4': l8,  'Sarima': l9, 'Sinu': l10})
    else:
        dataset = pd.DataFrame({'Constant': l12_n, 'LSTM1': l1_n, 'LSTM2': l2_n, 'LSTM3': l3_n, 'LSTM4': l4_n, 'Seq2Seq1': l5_n, 'Seq2Seq2': l6_n, 'Seq2Seq3': l7_n, 'Seq2Seq4': l8_n,  'Sarima': l9_n, 'Sinu': l10_n})
    thisfig = plt.figure(figsize=(6,4.5))

    sns.set(style="whitegrid")


    ax = sns.boxplot(data=dataset, showfliers = False)
    if normal == 1:
        plt.ylabel('RMSE (Distribution in Quartiles: For 310 Signals)')
    else:
        plt.ylabel('n2RMSE (Distribution in Quartiles: For 310 Signals)')
    plt.title(plotTitle)
    thisfig.savefig(plotName, bbox_inches='tight')

    plt.show()

showPlot(0,'1MonthNormalCon2.pdf','Prediction of 1 Month (Seasonal)',0)
#showPlot(4,'5Month.pdf','Prediction of 5 Month')
showPlot(8,'9MonthNormalCon2.pdf','Prediction of 9 Month (Seasonal)',0)
showPlot(0,'1MonthNotNormal2.pdf','Prediction of 1 Month (Seasonal)',1)
#showPlot(4,'5Month.pdf','Prediction of 5 Month')
showPlot(8,'9MonthNotNormal2.pdf','Prediction of 9 Month (Seasonal)',1)
