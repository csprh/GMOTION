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

    rmses = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]

    dataset = pd.DataFrame({'LSTM1': l1, 'LSTM2': l2, 'LSTM3': l3, 'LSTM4': l4, 'LSTM5': l5, 'LSTM6': l6, 'LSTM7': l7, 'LSTM8': l8,  'Sarima': l9, 'Sinu': l10})

    thisfig = plt.figure(figsize=(12,8))

    sns.set(style="whitegrid")


    ax = sns.boxplot(data=dataset, showfliers = False)
    plt.ylabel('RMSE (Distribution in Quartiles: For 210 Signals)')
    plt.title(plotTitle)
    thisfig.savefig(plotName, bbox_inches='tight')

    plt.show()

showPlot(0,'1Month.pdf','Prediction of 1 Month')
showPlot(4,'5Month.pdf','Prediction of 5 Month')
showPlot(8,'9Month.pdf','Prediction of 9 Month')