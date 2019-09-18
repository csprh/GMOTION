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
loadInterpLocation = 0;
thisGPS  = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)


cubeLenLX = 0.0075;
cubeLenLY = 0.0125;

Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);


if loadInterpLocation == 0
    interpLocation = interpRegion_use(frameName,cubeLenLX, cubeLenLY, Latitude0S, Longitude0S);
    save interpLocation interpLocation;
else
    load interpLocation;
end

signal1D= interpLocation.outcdTSmooth;

lat2 = interpLocation.lat2; lon2 = interpLocation.lon2;
plot(lon2,lat2,'.','color',[0.6,1,0.6]);

daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);


for ii = 1:size(signal1D,1)
    this_signal1D = signal1D(ii,:);

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
