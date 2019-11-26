function getSeasonalData
%% Function that 
%% 1. extracts data from satsense frame
%% 2. finds the data that surrounds a datapoint of interest
%% 3. interpolates missing data
%% 4. get's indices for the data indentified as seasonal (5, 6 and 7 sigma)
%
% USAGE:
%   getSeasonalData
% INPUT:
%   -
% OUTPUT:
%   save output to matlab file
% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill July 2019

clear; close all;

frameName = '030A_03647_101313-vel.h5'; %Satsense data
loadInterpLocation = 1;
choosePos = 1; %Normanton coal fields
gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [53.785766, -2.953747]; %New preston road fracking site
gps2 = [54.136083, -1.523892]; %Ripon
gps3 = [53.777 -3.035]; %BLAP
gps4 = [53.800 -1.664]; %LEED

gps = [gps0 ;gps1 ;gps2 ;gps3; gps4];

cubeLenL = 0.01;

thisGPS = gps(choosePos,:);
Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);


if loadInterpLocation == 0
    interpLocation = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S);
    save interpLocation interpLocation;
else
    load interpLocation;
end

signal1D_0= interpLocation.outcd;
signal1D_1= interpLocation.outcdFilt;
signal1D_2= interpLocation.outcdAPS;
signal1D_3= interpLocation.outcdTSmooth;

lat2 = interpLocation.lat2; lon2 = interpLocation.lon2;
plot(lon2,lat2,'.','color',[0.6,1,0.6]);

daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);


for ii = 1:size(signal1D_3,1)
    this_signal1D = signal1D_3(ii,:);

    % Remove 3rd degree polynomial trend
    opol = 3;
    t = 1:length(this_signal1D);
    [p,s,mu] = polyfit(t,this_signal1D,opol);
    f_y = polyval(p,t,[],mu);
    this_signal1D = this_signal1D - f_y;
    
    % Remove bit of the time series that makes it a full number of years
    fullLength = length(this_signal1D);
    thisRem = rem(fullLength, lagAC);
    this_signal1D = this_signal1D(thisRem:end); 

    % Find autocorrelation
    acf = autocorr(this_signal1D,lagAC);
    arrayAC(ii) =  abs(acf(lagAC+1));
end

% Obtrain 5,6 and 7 sigma bounds on the autocorrelation
numStds = 5; [~,~,bounds5] = autocorr(this_signal1D,lagAC,[],numStds);
numStds = 6; [~,~,bounds6] = autocorr(this_signal1D,lagAC,[],numStds);
numStds = 7; [~,~,bounds7] = autocorr(this_signal1D,lagAC,[],numStds);

arrayACInd5 = arrayAC>bounds5(1);
arrayACInd6 = arrayAC>bounds6(1);
arrayACInd7 = arrayAC>bounds7(1);

interpLocation.arrayACInd5 = arrayACInd5;
interpLocation.arrayACInd6 = arrayACInd6;
interpLocation.arrayACInd7 = arrayACInd7;
interpLocation.arrayAC = arrayAC;

save interpLocation interpLocation;
