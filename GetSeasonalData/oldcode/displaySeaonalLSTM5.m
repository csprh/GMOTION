function displaySeaonalLSTM5
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

load interpLocation
lon_1 = interpLocation.lon2;
lat_1 = interpLocation.lat2;

thisLon1 = lon_1(round(length(lon_1)/2));
thisLat1 = lat_1(round(length(lat_1)/2));

for ii = 1: length(lon_1)
   thisLon2 = lon_1(ii);
   thisLat2 = lat_1(ii); 
   
   [arclen,az] = distance(thisLat1,thisLon1,thisLat2,thisLon2);
   thisLen(ii) = arclen;
end

[Y,I] = sort(thisLen);

lons = lon_1(I(1:8));
lats = lat_1(I(1:8));

plot(lon_1,lat_1,'+c','MarkerSize',3);
hold on;
plot(lons,lats,'+r','MarkerSize',5);
plot(thisLon1,thisLat1,'*k','MarkerSize',6);



legend('Other Signals', 'Local Signals', 'Considered Signal');
addpath('..');

%plot_google_map('MapScale', 1, 'APIKey','AIzaSyCwSe-kkMTqCkG7jXXhIEgpLv8F5xAZi7U', 'Maptype', 'hybrid');
%plot_google_map('MapScale', 2, 'ShowLabels', 0, 'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
plot_google_map('MapScale', 1,  'ShowLabels', 0,'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');

