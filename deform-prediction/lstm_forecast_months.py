#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed July 24 16:38:10 2019

University of Bristol: Digital Environment and Dept. of Computer Science

@author: Dr. Víctor Ponce-López
"""

import pandas
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#from keras.layers import BatchNormalization
#from keras.layers import ConvLSTM2D
#from keras.layers import Conv3D
from keras import backend as be
from series_to_supervised import series_to_supervised
from utils import calcErr,plotPredictions,normbygroup
from reset_keras import reset_keras
import pickle
#from visualiseForecast import visualiseForecast
    
useGps = False
if useGps:
    ################# LOAD GPS data #############################
    datafiles = ['LEED_east', 'LEED_LOS', 'LEED_vert', 'LEED_north']
    #datafiles = ['HOOB_east', 'HOOB_LOS', 'HOOB_vert', 'HOOB_north']
    signals = ['east', 'LOS', 'vert', 'north'];    s = 1; 
    datafile = datafiles[s]
    dataset = pandas.read_csv(datafile, header=None, usecols=[0,1], delim_whitespace=True, engine='python')   # select date,y(m)
else:    
    ################# LOAD InSAR data #############################
    #datafiles = ['DARE_interp_may15-dec18_unfiltered.txt', 'DARE_interp_may15-dec18_Filt.txt', 'DARE_interp_may15-dec18_APS.txt', 'DARE_interp_may15-dec18_TSmooth.txt'];
    datafiles = ['Normanton_interp1day_may15-dec18_unfiltered.txt', 'Normanton_interp1day_may15-dec18_Filt.txt', 'Normanton_interp1day_may15-dec18_APS.txt', 'Normanton_interp1day_may15-dec18_TSmooth.txt'];
    #datafiles = ['Normanton_interp6day_may15-dec18_unfiltered.txt', 'Normanton_interp6day_may15-dec18_Filt.txt', 'Normanton_interp6day_may15-dec18_APS.txt', 'Normanton_interp6day_may15-dec18_TSmooth.txt'];
    #datafiles = ['Normanton_orig_may15-dec18_unfiltered.txt', 'Normanton_orig_may15-dec18_Filt.txt', 'Normanton_orig_may15-dec18_APS.txt', 'Normanton_orig_may15-dec18_TSmooth.txt'];
    ##datafiles = ['Leeds_interp1day_may15-dec18_unfiltered.txt', 'Leeds_interp1day_may15-dec18_Filt.txt', 'Leeds_interp1day_may15-dec18_APS.txt', 'Leeds_interp1day_may15-dec18_TSmooth.txt'];
    filterlevels = ['unfiltered', 'Filt', 'APS', 'TSmooth'];     fl = 3
    datafile = datafiles[fl]
    ## locations with highest seasonality: 4626,4280,3013,3353,4279,3835,5496,2805 -- previous tested signals: 4058,4061,8062,13113,4062,4060  --- non-seasonal: 5158,5200 up trendfor Normanton; 12994 for Leeds
    #dataset = pandas.read_csv(datafile, header=None, usecols=[4626], engine='python') 
    dataset = pandas.read_csv(datafile, header=None, usecols=[4626,4280], engine='python') 
    #dataset = pandas.read_csv(datafile, header=None, usecols=[4626,4280,3013,3353,4279,3835,5496,2805], engine='python') 
    #dataset = pandas.read_csv(datafile, header=None, 
    #                          usecols=[4626,4280,3013,3353,4279,2158,10771,13268,3835,5496,12298,2267,3834,10835,13245,13244,2384,2266,2805,8478,2750,13277,9851,2383,10586,11913,9059,13213,8477,12581,3420,4469,4295,2028,13215,12302,3660,6616,651,13243,12172,8169,8315,7998,11152,4307,4164,9793,12300,12171,11914,2163,789,3863,3080,11172,5300,3960,12582,2268,3833,12301,5804,11834,2165,10770,4425,3655,12389,2847,3860,5230,3494,2159,2065,8290,168,8170,8317,12182,11521,12384,11084,11173,12504,4588,5326,4728,2269,8165,2987,5092,2826,5505,4929,7910,4992,7997,4165,408,13242,8289,4911,13214,4587,790,13212,2945,4293,8274,555,13334,8316,4269,5506,2164,8168,5297,2726,10111,4353,11622,12399,788,453,12385,12505,8324,3751,4415,7192,4162,5228,2692,2162,3573,8307,5344,724,2157,652,13241,11962,13211,3354,3916,2380,3861,2166,8167,4426,8322,11083,11326,5593,8474,4422,726,7912,9726,11094,11153,3743,4054,727,8381,6217,11961,5490,7994,8166,11944,8326,3030,2160,12506,11943,12383,5598,11830,5091,5188,8325,10666,4060,2072,7913,7909,12289,3493,5807,12185,2029,12186,4123,3744,725,11231,7758,4294,410,3746,5111,142,11951,2270,4407,222,2944,9794,5185,10535,5292,3654,10476,12187,5604,8321,12175,11611,12550,4542,8853,8327,4163,6139,3961,839,6024,3657,2154,4453,2592,3753,3469,8085,8320,8164,12286,3742,8511,3658,5492,10651,12285,5611,11013,655,2660,5095,4437,13246,8377,4058,5594,12495,9692,3752,4928,12057,4586,1669,4061,170,11949,8488,11492,4412,12400,6218,4647,6820,2654,5612,13286,6176,10869,6039,5187,8357,96,9640,8092,5094,8318,2071,8254,6208,8308,4416,2690,2156,12516,5741,4547,3732,4020,5412,4062,5699,12077,12340,12995,653,12392,8084,4746,1658,1659,12994,2025,8240,10065,6933,4021,5515,9556,409,3748,7908,5583,4288,1354,656,12725,11743,4912,2265,10453,3652,6054,3651,5507,12996,10068,9555,8091,9772,12401,13267,11704,141,5066,3750,3933,8849,12306,5451,6219,806,5119,3572,6312,12982,3747,13333,7600,12176,4296,9852,4169,10753,8058,12353,6833,3740,5229,11208,10235,5504,6134,11537,2749,5491,12191,8086,9946,6173,12450,13050,4286,3656,10684,9422,2201,10365,2350,1471,8666,4864,10543,8496,10692,5426,6260,5722,654,11282,371,5584,12184,7996,5871,12390,12129,6046,10348,1340,221,11948,12291,12655,9829,4823,2528,10982,3938,13174,2765,3959,3662,10561,7083,3862,486,5695,10131,2066,4000,9543,12983,4630,11274,7870,979,4408,12287,8257,5600,12062,3942,3892,6221,4548,5595,12078,5692,11684,13126,2883,3474,1774,787,8642,3653,4914,8163,12084,9230,8245,2026,8243,10019,2989,3358,2832,4669,9268,12694,485,8246,5694,2494,6813,11501,4826,11870,407,1655,3661,10611,2635,12174,12267,4283,5873,4023,13302,12085,2602,4411,8244,3452,4171,10624,12810,12728,6385,7754,8256,136,11085,5348,9736,8540,10737,10697,3749,4178,4379,10282,8351,5614,8088,9474,12404,2070,10054,2233,4111,5112,8258,1242,8648,11286,8090,6596,223,8758,1698,3659,9716,8087,8252,2593,8429,405,4290,10196,5592,10669,2161,2232,10694,8242,554,3027,1554,2187,12081,6133,12290,3236,3741,12173,3982,8162,10538,6494,5089,600,723,10667,7911,12226,3829,6056,3911,7826,10616,10605,1501,3476,10613,11287,10265,9421,12355,8502,9133,4580,10110,8106,10195,6805,3922,12496,12247,9414,3058,5700,7978,2263,4926,9913,4518,13301,2981,13175,12903,6388,9863,1940,4579,4439,10996,6751,2414,4232,318,4638,11950,6336,11865,3647,10548,4994,12060,3099,10122,9538,8142,9159,3762,5448,344,8652,5615,11386,6383,66,6753,11396,10551,5436,7478,4300,11770,8861,12087,12940,10066,5186,3029,10161,3952,8495,547,495,5701,5231,10905,12190,7437,4918,4285,12566,5613,5799,2630,12,1862,11564,6380,10435,8161,3468,10132,1564,2537,5597,3930,2381,5691,12058,406,9945,8255,10750,11476,5331,3817,11154,846,1045,4396,7185,1784,11676,8171,8173,7907,12216,12501,2069,4642,3665,6318,7238,11682,431,8470,12240,2261,6932,12556,8475,4022,7993,8035,12341,319,5620,9990,10534,10807,11983,3402,6384,12080,1661,3470,9664,9361,1580,9860,6527,3921,8248,496,11883,7757,7076,8247,6337,5602,10456,9888,5693,2062,9267,5688,8459,4442,12079,591,12904,4417,8653,11833,8175,11879,4048,8260,10880,9875,3735,12845,10612,10042,3929,9446,2541,9820,1551,609,8428,2315,10610,10691,299,3931,13319,3571,10481,10421,5080,12442,12669,10937,6138,2487,6832,9629,12402,10843,5589,12049,12394,8064,5812,9587,234,8848,8259,3434,4913,9908,9002,2707,11473,12284,5361,7977,10762,12325,8981,7196,11880,8422,11739,10983,8251,8241,10854,8651,8450,566,9060,8722,9874,9401,12188,4782,3648,10314,10754,10060,6042,5346,3739,494,7756,2493,6349,2316,10064,4590,10614,8593,8493,12416,10468,8328,7325,11485,4308,4990,12061,6292,1397,8237,5237,8160,10455,5072,1052,10124,8423,7682,3899,10424,1083,6561,11567,8382,10288,8700,10537,10984,9771,7861,10617,97,12101,6817,169,9420,8311,9590,5411,1870,11411,11620,4519,5107,10857,3855,11054,461,4024,12906,4996,10044,3064,4378,8350,12268,2597,8556,5710,2661,4533,5720,7972,13051,9300,10062,404,12621,10977,5221,6204,13320,7563,3939,9810,8795,6707,10428,8725,3620,7724,11762,13007,8724,801,8424,3854,5090,6724,10073,3745,9837,11095,8178,5498,483,8158,10827,5416,10615,8177,10200,11580,11216,9615,2432,6222,4898,5216,755,11075,12086,9792,9798,13058,8461,10584,10057,4717,8295,5696,1939,1218,4298,12279,12668,5687,5526,3215,10239,6861,9161,3496,8319,9491,12494,498,1652,3028,10517,2689,1338,462,10079,4438,2490,8249,2803,10784,8622,6960,2489,4756,11581,3057,12257,11369,8420,3500,12386,1300,3957,11849,12761,5510,10547,11963,4148,9589,5222,8063,527,1192,11436,11460,12165,6171,7039,9849,5323,3943,10075,6316,9768,12524,4515,9872,5529,2615,6243,6818,6096,2328,6338,5918,9354,9997,8356,7803,5256,10910,5113,8589,12688,12618,5430,5944,10858,5599,10797,8174,4829,10774,12667,8323,5868,7827,12639,8421,8157,12523,9088,8141,12576,510,12714,9630,6394,8896,1977,5372,11562,8492,10106,9205,5725,5006,7324,5060,10121,1653,7886,2876,13084,10037,7305,3395,6519,6382,10003,7750,9282,7794,12579,11871,613,7971,11138,1339,13239,10261,2896,3650,4175,5274,4721,6736,8369,12410,3554,9821,10359,2618,4991,1082,5219,5061,8466,12941,1439,2565,5390,10856,2296,11472,8674,9160,2248,9628,4091,1044,10519,1169,8250,9078,4644,10356,5732,2691,8379,8661,12027,4512,12541,4693,9270,8723,5533,444,11966,10562,7678,11056,458,4816,12899,3371,6991,6063,4139,10283,7672,4071,8465,6927,8062,8528,1145,8373,684,4373,1451,5031,7743,4170,5499,6931,10386,3195,9977,8868,1178,10735,10906,4658,9550,4801,4006,3435,2419,2243,6053,8792,2421,10313,7999,4849,7751,4496,1665,11960,4287,11565,12600,9529,2249,9178,6926,10208,13121,850,6291,3738,7059,11127,10619,2506,8159,2655,7292,8364,1552,2725,7351,590,9969,2666,3437,1375,6205,9408,10215,11068,4822,4085,7676,12146,13339,4305,5958,6709,2274,6725,10533,3059,10830,473,5587,9926,158,12809,4493,7812,2982,2093,10539,7837,10826,10683,5919,8132,1275,7661,12599,2089,2625,10986,11397,297,3901,10477,12596,9077,8462,9233,5851,11462,4608,7170,3912,9362,10746,2108,5144,8417,11364,9574,2607,12167,6313,1323,9588,12723,7377,9668,10275,12280,6008,9426,9699,2370,10163,276,6041,13288,3891,8285,11629,11041,12990,4013,8780,11030,3265,4993,9912,8378,12082,6427,12893,1253,6866,8645,6137,11767,12139,12544,4544,8457,2988,5624,4808], 
    #                          engine='python') 

#### show data
dataset, datafile
    
#for i in range(dataset.shape[1]):
    # set displacement values to milimeters
    #k = np.random.randint(dataset.shape[1])
    #values = dataset[k].values[:,1]*1000 if useGps else dataset[k].values

values = dataset.values[:,1]*1000 if useGps else dataset.values
print(values.shape)

# plot displacement values in milimeters
plt.close()
plt.figure(figsize=(12,8))
plt.plot(values)
ndates = len(values)
plt.xlabel('Time blocks of %i dates per location' % (ndates))
plt.ylabel('Displacement (mm)')
plt.title('Deformation MAY\'15 - DEC\'18')
plt.show()

# select range of dates for years between MAY 2015 and DEC 2018
daysObs = 4#30          # set to 30 or 4 with or without interpolation, respectively
test_size = 60#365      # set to 365 or 60 with or without interpolation, respectively
nMonths = int(ndates/daysObs)+1   # number of total months in the data
print("%i total months" % (nMonths))
months = 9          # number of previous months to learn from
predMonths = 9      # number of months to forecast
seed = 4            # number of the starting random seed
nfeatures = 1       # number of different type of features (1: displacements only)
testSeqIdx = -1      # index of the target test sequence (-1: test ALL sequences)
savePath = '1D_1D/noreframe/'

assert predMonths*daysObs <= test_size

# specify columns (locations) to consider and to plot
groups = range(values.shape[1])
# plot each location
plt.close()
plt.figure(figsize=(12,30))
for group in groups:
    plt.subplot(len(groups), 1, group+1)
    #plt.plot(values[group*ndates:(group+1)*ndates])
    plt.plot(values[:,group])
    #plt.ylim(-15,15)
plt.show()

values, scaled, scalerCD = normbygroup(dataset, ndates, values, nfeatures, useGps)

print(scaled.shape)
np.min(scaled[:,0]),np.max(scaled[:,0])   # check data is normalised within range [0,1]

for pred in range(predMonths, predMonths+1):    # change to range(1, predMonths+1) to run and save several sizes of future observations
    # define temporal windows of past and future observations
    
    look_forward = daysObs*pred; 
    
    if testSeqIdx >= 0:
        test_y = values[-test_size:-test_size+look_forward, testSeqIdx]  # future observations of the target test sequene        
    
    errs, preds = [], []
    rmse = np.zeros(shape=(scaled.shape[1],3)) if testSeqIdx < 0 else np.zeros(shape=(1,3))
    for seed in range(4,7):       # run experiments with multiple random seeds
        plt.close()
        
        np.random.seed(seed)      # set the random seed
        
        # Initialise Scores
        trainPredict, testPredict = [], []
        for month in range(months,months+1):    # maximum period of 3 years observed
            print("Learning from previous %i months and %i monthly observations at random seed %i with architecture %s..." % (month, pred, seed, savePath))
            
            # define number of training samples
            look_back = month*daysObs
            
            # Multiple Lag Timesteps 
            ############################################################################################################################################
            ########################## https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ ############################
            ############################################################################################################################################

            ### 1D Architecture
            s = np.concatenate(np.concatenate(scaled[:-test_size,:],axis=1)).reshape((ndates-test_size)*values.shape[1],1)     # Concat 1D
            #s = scaled[:-test_size,:].reshape((ndates-test_size)*values.shape[1],1)      # Point-by-point concat 1D
            ### Multidimensional Architecture
            #s = scaled[:-test_size,:]    # Reshape 2D   
            
            # frame as multivar supervised learning 
            train = series_to_supervised(s, nfeatures, look_back, look_forward)
            print(train.head(), train.shape)
            
            n_futObs = look_forward * nfeatures * s.shape[1]
            n_obs = look_back * nfeatures * s.shape[1]
                            
            # Multiple Lag Timesteps 
            ############################################################################################################################################
            ########################## https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ ############################
            ############################################################################################################################################
            # This will modify the LSTM' input and output shapes to be [samples, observations, features times future observations]
            
            # Set values into train and test sets
            train = train.values
            train_y = train[:, -n_futObs:] if s.shape[1]==1 else train[:, -n_futObs+testSeqIdx::s.shape[1]]    # future observations of the target training sequene
            
            ### 1D Architecture
            #s = np.concatenate(np.concatenate(scaled[-test_size-n_obs:-test_size,:],axis=1)).reshape(n_obs*values.shape[1],1)     # Concat 1D
            #s = scaled[-test_size-n_obs:-test_size,:].reshape(n_obs*values.shape[1],1)      # Point-by-point concat 1D
            ### Multidimensional Architecture
            #s = scaled[-test_size-n_obs:-test_size,:]    # Reshape 2D   
            
            # frame as multivar supervised learning 
            #test = series_to_supervised(s, nfeatures, look_back-1, 1)   # include last sample before 'today'
            #print(test.head(),test.shape)
            #test = test.values
            
            #######################################################################################################################################
            ### 1D Architecture - with last parts of training divided proportionally into 1D
            test = np.concatenate(np.concatenate(scaled[-test_size-int(n_obs/scaled.shape[1]):-test_size,:],axis=1))  # Concat 1D
            #test = scaled[-test_size-int(n_obs/scaled.shape[1]):-test_size,:]   # Point-by-Point concat 1D 
            test = test.reshape(1,test.shape[0])    # Concat 1D
            n_obs = test.shape[1]         # verification and readjustment of inputs, if necessary            
            ### Multidimensional Architecture - with whole training sequences 
            #test = np.concatenate(np.concatenate(scaled[-test_size-n_obs:-test_size,:],axis=1))    # Concat & Point-by-point Reshape 2D (Shrink)
            #test = scaled[-test_size-n_obs:-test_size,:]          # Reshape 2D 
            #######################################################################################################################################
            
            if testSeqIdx > 0: test = scaled[-test_size-n_obs:-test_size,testSeqIdx].reshape(1, n_obs)     # Target sequence only
            
            train_X = train[:, :n_obs]; test_X = test[:, :n_obs]
            # This will modify the LSTM' input and output shapes to be [samples, observations, features times future observations]
            test_X = test_X.reshape((test_X.shape[0], n_obs, nfeatures))
            train_X = train_X.reshape((train_X.shape[0], n_obs, nfeatures))
            print(train_X.shape, train_y.shape)
            
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.5 
            be.tensorflow_backend.set_session(tf.Session(config=config))
            
            # # design network Architecture 
            model = Sequential()
            
            # 1D_1D architecture -- values empirically tested in several signals
            model.add(LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
            model.add(Dropout(0.1))      # 0.605 (0.25,0.25,0.25); 0.899 (0.25,0.1,0.25); 1.369 (0.5,0.1,0.1); # 0.917 (0.25,0.5,0.25); 0.647 (0.1,0.25,0.25)
            model.add(LSTM(128, return_sequences=False))
            model.add(Dropout(0.2))      # 0.5 (0.1,0.2,0.25); 0.507 (0.1,0.1,0.25); 0.853  (0.1,0.1,0.20); 0.642 (0.1,0.15,0.25); 0.901 (0.1,0.15,0.20); 
            model.add(Dense(128))
            model.add(Dropout(0.25))     # 0.877 (0.25,0.25,0.1); 1.2 (0.25,0.1,0.1); 1.071 (0.1,0.1,0.1); 0.655 (0.1,0.25,0.25); 0.855 (0.1,0.25,0.25); 0.576 (0.15,0.2,0.25)
            model.add(Dense(train_y.shape[1]))
            
            # ConvLSTM2D architecture   # for Multidimensional input sequences
#            model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                   input_shape=(None, train_X.shape[1], train_X.shape[2], 1),
#                   padding='same', return_sequences=True))
#            model.add(BatchNormalization())#            
#            model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                               padding='same', return_sequences=True))
#            model.add(BatchNormalization())
#            model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                               padding='same', return_sequences=True))
#            model.add(BatchNormalization())#            
#            model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                               padding='same', return_sequences=True))
#            model.add(BatchNormalization())#            
#            model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                           activation='sigmoid',
#                           padding='same', data_format='channels_last'))
            
            # old architecture
            ##model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            ##model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))    # 50 neurons, n_obs and nfeatures timesteps as inputs
            #model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
            ##model.add(Dropout(0.5))
            #model.add(Dense(train_y.shape[1]))                             # n_futObs timesteps as outputs

            #model.compile(loss='mse', optimizer='adadelta')   # for ConvLSTM2D architecture
            model.compile(loss='mse', optimizer='adam')
            
            # fit network
            history = model.fit(train_X, train_y, epochs=n_obs*60, batch_size=values.shape[1]*10, verbose=1, shuffle=False)
            
            # SAVE model
            #pickle.dump(model, open(savePath+'model-seed'+str(seed)+'-'+str(pred)+'-preds-'+str(month)+'obsmonth.pkl','wb'))
            #pickle.dump(history, open(savePath+'history-seed'+str(seed)+'-'+str(pred)+'-preds-'+str(month)+'obsmonth.pkl','wb'))
            
            # show learning process
            #plt.close()
            #plt.plot(history.history['loss'], label='train')
            #plt.plot(history.history['val_loss'], label='test')
            #plt.xlabel('Epoch')
            #plt.ylabel('Loss')
            #plt.legend()
            #plt.show()    
            
            # save training predictions
            #trainPredict.append(scalerCD.inverse_transform(model.predict(train_X))[:,0])
            
            # Make predictions, save error and plot 
            yhat = model.predict(test_X)
            if testSeqIdx < 0:   # Evaluate every target test sequences 
                for i in range(scaled.shape[1]):
                    test_y = values[-test_size:-test_size+look_forward, i]
                    rmse[i,seed-4], inv_yhat, inv_y = calcErr(yhat, test_y, scalerCD)
                    plotPredictions(values[:,i], ndates-test_size, str(pred), inv_yhat, inv_y, '')                    
            else:                # Evaluate a given target test sequence 
                rmse[0,seed-4], inv_yhat, inv_y = calcErr(yhat, test_y, scalerCD)
                plotPredictions(values[:,testSeqIdx], ndates-test_size, str(pred), inv_yhat, inv_y, '')
            
            del model, history
            be.clear_session(); reset_keras()   # Reset Keras Session
            
            print('RMSE for every signal: ')
            print(*rmse[:,seed-4], sep = ", ")              
            
            
        #visualiseForecast(plotFit, earlySeason, test_size, values, ndates, months, daysObs, n_futObs, trainPredict, testPredict, rmse)
        
        # generate 3 sets of random means and confidence intervals to plot
        preds.append(inv_yhat)
        for i in range(rmse.shape[0]):
            errs.append(rmse[i,:]); 
        
    pickle.dump(preds, open(savePath+'preds'+str(pred)+'.pkl', 'wb'))
    # SAVE errors and predictions
    for i in range(1,rmse.shape[0]+1):
        pickle.dump(errs[i-1], open(savePath+'errs'+str(pred)+'_SIG'+str(i)+'.pkl', 'wb'))
    

print('\n100%% done!')



####################################################################################
# tail code, just for reading stored results and generate different visualisations #
####################################################################################


## load data -- test 
nx = [savePath[:-1]]
months = 9; pred = 9
#nx = [savePath]; 

e,p = [],[]
for pred in range(1,predMonths+1):
    p.append(pickle.load(open(nx[0]+'/preds'+str(pred)+'.pkl', 'rb')))
savePath = nx[0]
nx = []

from plotConfidentInt import plot_mean_and_CI, plotConfidentInt

# Forecast different periods of y -- plot predictions over all signals sorted by seasonality index 
meanOnly = False
testSeqIdx = [4626,4280,3013,3353,4279,3835,5496,2805]
for i in testSeqIdx:
    inv_y = dataset[i].values[-test_size:-test_size+n_obs];
    tr = dataset[i].values[:-test_size].reshape(1,ndates-test_size)[0,:];
    plotConfidentInt([], p, inv_y, tr, [], daysObs, pred, nx, 'all', meanOnly, '')
    s = savePath+'/preds'+str(pred)+'_all_SIG'+str(testSeqIdx.index(i)+1)+'-'+str(len(testSeqIdx))+'.png' if len(p)==1 else savePath+'/preds1-'+str(pred)+'_all_SIG'+str(testSeqIdx.index(i)+1)+'-'+str(len(testSeqIdx))+'.png'
    #plt.show()
    plt.savefig(s)
    if testSeqIdx.index(i) == 0:
        plotConfidentInt([], p, inv_y, tr, [], daysObs, pred, nx, 'superzoom', meanOnly, '')
        s = savePath+'/preds'+str(pred)+'_zoom_SIG'+str(testSeqIdx.index(i)+1)+'.png' if len(p)==1 else savePath+'/preds1-'+str(pred)+'_zoom_SIG'+str(testSeqIdx.index(i)+1)+'.png'
        #plt.show()
        plt.savefig(s)
plt.close()   

## Forecast a fixed period y with different observed periods x
#nx = ['Nx1', 'Nx6', 'Nx9', 'Nx12']; 
#nx = ['Nx9/signal2', 'Nx9/twosignals']; 
#p,e = [],[]
#nx = [nx]
#for m in range(0,len(nx)):
#    e.append(pickle.load(open(nx[m]+'/errs'+str(pred)+'.pkl', 'rb')))
#    

## Calc Errs on another test signal 
import math
from sklearn.metrics import mean_squared_error
nx = [savePath]; e = []; seed=0; 
for j in testSeqIdx:#range(values.shape[1]):#testSeqIdx:
    for i in range(predMonths-len(p)+1,predMonths+1):
        test = dataset[j].values[-test_size:]
        #test = values[-test_size:, j]
        look_forward = daysObs*i; test_y = test[:look_forward]
        inv_y = test_y
        #inv_y = scalerCD.inverse_transform(test_y.reshape(1,len(test_y)))[0,:]
        errs = []
        for seed in range(0,len(p[0])):
            inv_yhat = p[i-(predMonths-len(p)+1)][seed]
            errs.append([math.sqrt(mean_squared_error(inv_y, inv_yhat))])
        pickle.dump(errs, open(nx[0]+'/errs'+str(i)+'_SIG'+str(testSeqIdx.index(j)+1)+'.pkl', 'wb'))
        e.append(errs)

#nx = ['Nx'+str(months)+'/old6signals/2D_concatpp2D/fold1', 'Nx'+str(months)+'/old6signals/2D_concatpp2D/fold2']; #nx = ['Nx'+str(months)+'/twosignals/1D_1D', 'Nx'+str(months)+'/twosignals/2D_2D', 'Nx'+str(months)+'/twosignals/2D_pp1D']
nx = [savePath]
nnx = []; e = []
for x in nx:
    for j in testSeqIdx:#range(values.shape[1]):#testSeqIdx:
        errs = np.empty(shape=(len(p), len(p[0])))
        for i in range(predMonths-len(p)+1,predMonths+1):
            err = np.array(pickle.load(open(x+'/errs'+str(i)+'_SIG'+str(testSeqIdx.index(j)+1)+'.pkl', 'rb')))
            if len(err.shape) > 1:
                errs[i-(predMonths-len(p)+1),:] = err[:,0]
            else: errs[i-(predMonths-len(p)+1),:] = err
        e.append(np.transpose(errs))
        nnx.append('SIG'+str(1+testSeqIdx.index(j)))
nx = nnx; pred=9

# plot errors over predicted months
if len(p) > 1:
    plotConfidentInt(e, [], inv_y, [], [], daysObs, pred, nx, 'superzoom', meanOnly, '')      # plot errors or no real seq.
    #plt.show()
    plt.savefig(savePath+'/errs1-'+str(pred)+'_SIG1-'+str(len(e))+'.png')

elif len(p) == 1:
    if len(e) > 2:
        # mean error plot 
        plt.figure(figsize=(12,8)); plot_mean_and_CI(np.mean(e,axis=1)[:,0], np.min(e,axis=1)[:,0], np.max(e,axis=1)[:,0], preds, '', '', False); 
        if e[0].shape[0] > 1:
            plt.title('Mean prediction errors and confidence intervals of all signals', fontsize=14)
        else:
            plt.title('Prediction errors of all signals', fontsize=14)
        plt.ylabel('Root Mean Square Error', fontsize=12)
        plt.xlabel('Signal ID', fontsize=12)
        plt.xticks(np.arange(len(e)), nx, fontsize=12)
        #plt.show()
        plt.savefig(savePath+'/errs'+str(pred)+'_SIG1-'+str(len(e))+'.png')
    
    # # mean error barplot
    plt.close()
    plt.figure(figsize=(12,8))
    plt.bar(np.arange(len(e)), np.mean(e,axis=1)[:,0], 0.5, yerr=np.std(e,axis=1)[:,0])
    #plt.bar(np.arange(pred), np.mean(e[0],axis=0), 0.5, yerr=np.std(e[0],axis=0))
    plt.ylabel('Root Mean Square Error', fontsize=12)
    plt.title('Mean error and Standard deviation for all signals with observed periods of '+str(pred)+' months', fontsize=12)
    plt.xticks(np.arange(len(e)), nx, fontsize=12)
    #plt.show()
    plt.savefig(savePath+'/barErrs'+str(pred)+'_SIG1-'+str(len(e))+'.png')
    
#from utils import plotTrainWindows
#plotTrainWindows(values, n_obs, t=9, nplots=5)
