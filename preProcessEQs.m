function preProcessEQs

clear; close all;
gebcoFilename = '/Users/csprh/seadas-7.4/DATA/GEBCO.nc';
filename = 'EQsUK2015-2018.xlsx';
[NUM,TXT,RAW] = xlsread(filename);

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
lon1D = ncread(gebcoFilename, '/lon'); 
lat1D = ncread(gebcoFilename, '/lat'); 

numOfPos = length(NUM);
numOfNeg = numOfPos*2;
nonSeaIndx = 0;
for ii=1: numOfPos
    outLat = RAW{ii,3};
    outLon = RAW{ii,4};
    outDat = RAW{ii,1};
    outML = RAW{ii,6};
    
    
    [~, centre_col] = min(abs(lon1D-outLon));
    [~, centre_row] = min(abs(lat1D-outLat));
    bathAt = ncread(gebcoFilename, '/elevation', [centre_col centre_row], [1 1]);
    isSea = bathAt <0;
    if isSea
        %plotm(outLat,outLon,'+b');
        outPrint = RAW(ii,10);
        
    else
        nonSeaIndx = nonSeaIndx + 1;
        %plotm(outLat,outLon,'+r');
        Longitude(nonSeaIndx) = outLon;
        Latitude(nonSeaIndx) = outLat;
        thisDate(nonSeaIndx) = x2mdate(outDat);
        ML(nonSeaIndx) = outML;
    end
end

distThresh = 50;
datThresh = 50;
ii = 0;
while ii < numOfNeg
  % generate random position
  % 
    ranLon = rand(1,1)*(lonLims(2)-lonLims(1)) + lonLims(1);
    ranLat = rand(1,1)*(latLims(2)-latLims(1)) + latLims(1);
    ranDat = rand(1,1)*(datLims(2)-datLims(1)) + datLims(1);

    
    [arclen,az] = distance(ranLat,ranLon,Latitude,Longitude);
    distkm = distdim(arclen,'deg','km');
    [thisMinDist ,thisMinIndx] = min(distkm);
    thisMinDat = abs(thisDate(thisMinIndx)-ranDat);
    ii
    if (thisMinDist< distThresh) && (thisMinDat < datThresh);
        continue;
    end
    
    [~, centre_col] = min(abs(lon1D-ranLon));
    [~, centre_row] = min(abs(lat1D-ranLat));
    bathAt = ncread(gebcoFilename, '/elevation', [centre_col centre_row], [1 1]);
    isSea = bathAt <0;
    if isSea 
        continue;
    end
    ii = ii+1;
    Longitude(ii+nonSeaIndx) = ranLon;
    Latitude(ii+nonSeaIndx) = ranLat;
    thisDate(ii+nonSeaIndx) = ranDat;
    ML(ii+nonSeaIndx) = 0;
end

for ii = 1: length(ML)
    outLat = Latitude(ii);
    outLon = Longitude(ii);
    if ML(ii) == 0
        plotm(outLat,outLon,'+r');
    else
        plotm(outLat,outLon,'+b');
    end
end

save UKEQs2016-2018 Longitude Latitude thisDate ML

