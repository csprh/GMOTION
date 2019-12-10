function getSeasonalData_AllMetrics
% Choose either All satsense data or Normanton
% Define Name (either All or Norm)
% Output seasonality Index
% 
%
% USAGE:
%   getSeasonalData_AllMetrics
% INPUT:
%   -
% OUTPUT:
%   save output to matlab file
% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill Dec 2019

clear; close all;

loadInterpLocation = 1; normanton = 1; doPlot = 0;

frameName = '../030A_03647_101313-vel.h5'; %Satsense data


thisGPS = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)

if normanton == 1
    cubeLenLx = 0.01;
    cubeLenLy = 0.01;
    interpLocationName = 'interpLocationNorm';
else
    cubeLenLx = 1000;
    cubeLenLy = 1000;
    interpLocationName = 'interpLocationAll';
end

Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);


if loadInterpLocation == 0
    interpLocation = interpRegionSmooth(frameName, cubeLenLx, cubeLenLy, Latitude0S, Longitude0S);
    save (interpLocationName, 'interpLocation');
else
    load (interpLocationName);
end

signal1D_3= interpLocation.outcdTSmooth;

if doPlot
    lat2 = interpLocation.lat2; lon2 = interpLocation.lon2;
    plot(lon2,lat2,'.','color',[0.6,1,0.6]);
end
daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);


for ii = 1:size(signal1D_3,1)
    this_signal1D = signal1D_3(ii,:);
    plot(this_signal1D,'r');hold on;
    % Remove 3rd degree polynomial trend
    opol = 3;
    t = 1:length(this_signal1D);
    [p,s,mu] = polyfit(t,this_signal1D,opol);
    f_y = polyval(p,t,[],mu);
    this_signal1D = this_signal1D - f_y;
    plot(this_signal1D,'b');
    
    % Remove bit of the time series that makes it a full number of years
    fullLength = length(this_signal1D);
    thisRem = rem(fullLength, lagAC);
    this_signal1D = this_signal1D(thisRem:end); 

    % Find autocorrelation
    acf = autocorr(this_signal1D,lagAC);
    arrayAC(ii) =  abs(acf(lagAC+1));
    [ccoef1, ccoef2] = fitSinusoidMetric(this_signal1D);
    arraySin(ii) = ccoef1;
    [' AC12='  num2str(arrayAC(ii)) ' ccoef2=' num2str(ccoef2)]
    
end

for ii = 1: 10
    [~,~,bounds] = autocorr(this_signal1D,lagAC,[],ii);
    outBounds(ii)  = bounds;
end

interpLocation.arrayAC = arrayAC;
interpLocation.arraySin = arraySin;
interpLocation.outBounds = arraySin;


save (interpLocationName, 'interpLocation');
