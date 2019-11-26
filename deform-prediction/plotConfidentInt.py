#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:11:06 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(1,mean.shape[0]+1), ub, lb,color=color_shading, alpha=.2)
    # plot the mean on top
    plt.plot(range(1,mean.shape[0]+1), mean, color_mean, label='Forecast (mean)')


def plotConfidentInt(errs, preds, y, y_te, tr, x_tr, month, plotType):
    mean0 = np.mean(preds, axis=0)
    ub0 = np.max(preds, axis=0)# + .5
    lb0 = np.min(preds, axis=0)# - .5

    # fill with entire signal
    if plotType == 'all' or plotType == 'zoom':
        mean0 = np.concatenate((np.full(len(tr), np.nan), mean0))
        ub0 = np.concatenate((np.full(len(tr), np.nan), ub0))
        lb0 = np.concatenate((np.full(len(tr), np.nan), lb0))

    # plot the data
    plt.close()
    if plotType == 'all':      # plot all data
        plt.plot(np.arange(1,len(tr)+len(y_te)+1), np.concatenate((tr,y_te)), label='Real Sequence')
        plt.plot(np.arange(1,len(tr)-len(y)+1), np.concatenate((np.full(len(tr)-len(x_tr)-len(y), np.nan), x_tr)), label='Training period', color='brown')
    elif plotType == 'zoom':   # focus plot on training period
        plt.plot(np.arange(len(tr)-len(x_tr)-len(y)+1,len(tr)-len(y)+1), x_tr, label='Training period', color='brown')
        plt.plot(np.arange(len(tr)-len(x_tr)-len(y)+1,len(tr)+len(y)+1), np.concatenate((np.full(len(x_tr)-1, np.nan), tr[-len(y)-1:], y)), label='Real Sequence')
    else:                      # zoom forecast region
        plt.plot(np.arange(1,len(y)+1), y, label='Real Sequence')
    plot_mean_and_CI(mean0, lb0, ub0, color_mean='orange', color_shading='orange')
    plt.legend(loc='best')

    plt.title('Mean and confidence intervals, learning from '+str(month)+' months')
    plt.tight_layout()
    plt.grid()
    plt.show()

    #plt.close()
    #plt.plot(yy, label='Real Sequence')
    #plt.plot(preds[0], label='Forecast1-'+str(1))
    #plt.plot(preds[1], label='Forecast2-'+str(1))
    #plt.plot(preds[2], label='Forecast3-'+str(1))
    #plt.legend(loc='best')
    #plt.show()

    plt.close()
    plt.boxplot(errs); plt.xticks([])
