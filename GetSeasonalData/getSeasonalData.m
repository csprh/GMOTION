function inFrameQ2

clear; close all;

latLims =  [52.1914  54.8680];   
lonLims =  [-5.3161  -0.7730];

gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [53.785766, -2.953747]; %New preston road fracking site
gps2 = [54.136083, -1.523892]; %Ripon
gps3 = [53.777 -3.035]; %BLAP
gps4 = [53.800 -1.664]; %LEED

gps = [gps0 ;gps1 ;gps2 ;gps3; gps4];

cubeLenL = 0.01;

thisGPS = gps(1,:);
Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);

frameName = '030A_03647_101313-vel.h5';

out = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S);

signal1D= out.outcdFilt;

daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);

threshAC0 = 0.5;threshAC1 = 0.4;
threshAC2 = 0.3;threshAC3 = 0.25;


for ii = 1:size(signal1D,1)
    if rem(ii,10)==0
        ii
    end
    this_signal1D = signal1D(ii,:);

    ac = autocorr(this_signal1D,lagAC);
    arrayAC(ii) =  abs(ac(lagAC));
end

arrayACInd0 = arrayAC>threshAC0;
arrayACInd1 = arrayAC>threshAC1;
arrayACInd2 = arrayAC>threshAC2;
arrayACInd3 = arrayAC>threshAC3;

this_signal1DInd0 = signal1D(arrayACInd0,:);
this_signal1DInd1 = signal1D(arrayACInd1,:);
this_signal1DInd2 = signal1D(arrayACInd2,:);
this_signal1DInd3 = signal1D(arrayACInd3,:);




