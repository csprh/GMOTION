function getAllEQCubesv1

clear; close all;

%load coast
%plotm(lat,long)

%start = [1 1 1]; 
%count = [68941 18482 1];
%lat1 = h5read('030A_03647_101313-vel.h5','/Latitude', start(1:2), count(1:2));
%cd1 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);

latLims =  [52.1914  54.8680];   
lonLims =  [-5.3161  -0.7730];
datLims = [datenum('11-May-2015') datenum('27-Dec-2018')];


worldmap([49 59],[-12 4]) 
geoshow('landareas.shp') 

load UKEQs2016-2018
for ii = 1: length(ML)
    outLat = Latitude(ii);
    outLon = Longitude(ii);
    if ML(ii) == 0
        plotm(outLat,outLon,'+r');
    else
        plotm(outLat,outLon,'+b');
    end
end


