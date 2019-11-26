#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import matplotlib.pyplot as plt
import numpy as np

# This function plots the predictions over the training and test (forecast) sets and reports the RMSE on testing

# Inputs:
#   test_size: the size of the test set is the last year period given by var test_size (default: 365 days)
#   values: the whole original signal 
#   ndates: total number of samples (dates) (default: 1327 days)
#   months: number of total months considered for the analysis
#   earlySeason: flag which indicates the starting point of the validation set either from the early stages of the previous year season or from the last stages of the previous season
#   plotfit: flag which indicates whether or not to plot the learned predictions
#   trainPredict, testPredict: predicted values for train and test sequences, respectively
#   timeRMSE: RMSE values for each month, main reported regression score
def visualiseForecast(plotfit, earlySeason, test_size, values, ndates, months, daysObs, trainPredict, testPredict, timeRMSE):
    
    if plotfit:
        if test_size == 365:       # 1 day interp
            if earlySeason:
                shifts0 =  [14, 15, 16, 17, 18, 20, 19, 17, 15, 13, 12];  
                shifts12 = [14, 15, 15, 13, 11, 9, 7, 5, 3, 1, 0];    
                shifts24 = [7, 5, 3, 1, -1, -3, -5];
                shiftsEarly = [(False,0), (False,0), (False,0), (False,0), (True,0.10), (True,0.10), (True,3.10), (True,4.10), (True,3.10), (True,2.10), (True,2.10)]
                shiftsEarly12 = [(False,0), (False,0), (True,0), (True,3), (True,6.10), (True,6), (True,6), (True,4.10), (True,3.10), (True,2.10), (True,2.10)]
                shiftsEarly24 = [(True,6), (True,9), (True,8), (True,7), (True,6.25), (True,6), (True,6)]
            else:
                shifts0 =  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11];
                shifts12 = [2, 3, 4, 5, 6, 7, 7, 5, 3, 1, 0]; 
                shifts24 = [2, 3, 3, 1, -1, -3, -5];
        else:                      # no interp
            if earlySeason:
                # proper shifts to be set
                shifts0, shifts12, shifts24 = [], [], []
            else:
                shifts0 =  [2, 4, 5, 6, 7, 8, 8, 9, 8, 8, 8];
                shifts12 = [2, 4, 5, 6, 7, 6, 5, 5, 3, 2, 1];
                shifts24 = [2, 2, 0, -2, -4, -6, -6];
    r_months = range(0,11) if months >= 12 else months
    for month in r_months:
        if plotfit:
            shift0, shift12 = shifts0[month], shifts12[month]
        monthSets = [month, month+12, month+24]
        plt.close()
        if month < 7:
            plt.title('Best forecasts learning from '+str(monthSets[0]+1)+', '+str(monthSets[1]+1)+' and '+str(monthSets[2]+1)+' previous months')
        else:
            plt.title(str(len(testPredict[month]))+' days forecasts learning from '+str(monthSets[0]+1)+', and '+str(monthSets[1]+1)+' previous months')
           
        plt.plot(values, label='Real Sequence', color='blue')
        plt.plot(np.concatenate((np.full(ndates-test_size+np.where(timeRMSE[month]==np.nanmin(timeRMSE[month]))[0][0], np.nan), testPredict[monthSets[0]][:,0])), label='Forecast-'+str(monthSets[0]+1), color='green')
        #+(test_size-testPredict[monthSets[0]].shape[0])
        if plotfit:
            delay = ndates-test_size-(monthSets[0]+shift0)*daysObs
            if earlySeason and shiftsEarly[month][0]:
                if earlySeason: delay2 = int(daysObs*(shiftsEarly[month][1]))
                plt.plot(np.concatenate((np.full(delay, np.nan), trainPredict[month][:-delay2,0], np.full(len(testPredict[month])*2, np.nan), trainPredict[month][-delay2:,0])), label='PredictedTr-'+str(monthSets[0]+1), color='fuchsia')
            else:
                plt.plot(np.concatenate((np.full(delay, np.nan), trainPredict[month][:,0])), label='Fit-'+str(monthSets[0]+1), color='fuchsia')
        if months > 12:
            plt.plot(np.concatenate((np.full(ndates-test_size+np.where(timeRMSE[monthSets[1]]==np.nanmin(timeRMSE[monthSets[1]]))[0][0], np.nan), testPredict[monthSets[1]][:,0])), label='Forecast-'+str(monthSets[1]+1), color='fuchsia')
            if plotfit:
                delay = ndates-test_size-(monthSets[1]+shift12)*daysObs
                if earlySeason and shiftsEarly12[month][0]:
                    if earlySeason: delay2 = int(daysObs*(1+shiftsEarly12[month][1]))
                    plt.plot(np.concatenate((np.full(delay, np.nan), trainPredict[monthSets[1]][:-delay2,0], np.full(len(testPredict[month])*2, np.nan), trainPredict[monthSets[1]][-delay2:,0])), label='Fit-'+str(monthSets[1]+1), color='orange')
                else:
                    plt.plot(np.concatenate((np.full(delay, np.nan), trainPredict[monthSets[1]][:,0])), label='Fit-'+str(monthSets[1]+1), color='orange')
            if months > 24 and month < 7:
                plt.plot(np.concatenate((np.full(ndates-test_size+np.where(timeRMSE[monthSets[2]]==np.nanmin(timeRMSE[monthSets[2]]))[0][0], np.nan), testPredict[monthSets[2]][:,0])), label='Forecast-'+str(monthSets[2]+1), color='yellow')
                if plotfit:
                    shift24 = shifts24[month]
                    delay = ndates-test_size-(monthSets[2]+shift24)*daysObs
                    if earlySeason and shiftsEarly24[month][0]:
                        if earlySeason: delay2 = int(daysObs*(1+shiftsEarly24[month][1]))
                        plt.plot(np.concatenate((np.full(delay, np.nan), trainPredict[monthSets[2]][:-delay2,0], np.full(len(testPredict[month])*2, np.nan), trainPredict[monthSets[2]][-delay2:,0])), label='Fit-'+str(monthSets[2]+1), color='brown')
                    else:
                        plt.plot(np.concatenate((np.full(delay, np.nan), trainPredict[monthSets[2]][:,0])), label='Fit-'+str(monthSets[2]+1), color='brown')                    
        plt.xlabel('Day')                          # use for the averaged CDs
        plt.ylabel('Cumulative Displacement')
        plt.legend(bbox_to_anchor=(0.9, 1))
        plt.show()
        plt.close()
            
        ##################### Plot errors #####################
        plt.title('RMSE of predicted future observations for each model')
        plt.ylim(0,10)
        plt.plot(timeRMSE[month], label='RMSE-'+str(month+1)+'prevMonths', color='green')
        if months > 12:
            plt.plot(timeRMSE[monthSets[1]], label='RMSE-'+str(month+13)+'prevMonths', color='fuchsia')
            if months > 24 and month < 7:
                plt.plot(timeRMSE[monthSets[2]], label='RMSE-'+str(month+25)+'prevMonths', color='yellow')

        plt.ylabel('Root Mean Square Error')
        plt.xlabel('Day')
        plt.legend()
        plt.show()