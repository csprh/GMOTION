function displaySeaonalData
%% Function that 
%% 1. loads data from interpLocation
%% 2. visualise selected data
%
% USAGE:
%   displaySeaonalData
% INPUT:
%   -
% OUTPUT:
%   -
% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill July 2019
clear; close all;
addpath('../');
load interpLocationNorm3

plotOut(interpLocation, interpLocation.arrayS,1);
figure;
plotOut(interpLocation, interpLocation.arraySin,2);

function plotOut (interpLocation, arrayAC, numOut)


[ACS, ACSInd] = sort(arrayAC);
len4 = round(length(ACS)/4);

thresh1 = ACS(len4);
thresh2 = ACS(len4*2);
thresh3 = ACS(len4*3);

ind1 = arrayAC<thresh1;
ind2 = arrayAC<thresh2&arrayAC>thresh1;
ind3 = arrayAC<thresh3&arrayAC>thresh2;
ind4 = arrayAC>thresh3;


lat_1 = interpLocation.lat2(ind1); 
lon_1 = interpLocation.lon2(ind1);
lat_2 = interpLocation.lat2(ind2); 
lon_2 = interpLocation.lon2(ind2);
lat_3 = interpLocation.lat2(ind3); 
lon_3 = interpLocation.lon2(ind3);
lat_4 = interpLocation.lat2(ind4); 
lon_4 = interpLocation.lon2(ind4);


hold on;


plot(lon_1,lat_1,'.y','MarkerSize',10);
plot(lon_2,lat_2,'.g','MarkerSize',10);
plot(lon_3,lat_3,'.r','MarkerSize',10);
plot(lon_4,lat_4,'.b','MarkerSize',10);
hold on;

if numOut == 1
%    subplot(1,2,1);
    title('STL Based Seasonality', 'FontSize',14);
else
%    subplot(1,2,2);
    title('Sinusoid Fitting Based Seasonality', 'FontSize',14);
end
legend({'Low Seasonality', 'Mid-Low Seasonality', 'Mid-High Seasonality', 'High Seasonality'}, 'FontSize',14);

addpath('..');
plot_google_map( 'Maptype', 'satellite', 'ShowLabels', 0,'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
linkaxes
set(gca,'XTick',[], 'YTick', [])


