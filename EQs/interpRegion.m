function out = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S)
%% Function that 
%% 1. locates satsense data surrounding a lat, lon location
%% 2. interpolates this data (for missing data points)

% USAGE:
%   out = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S)
% INPUT:
%   frameName: Name of satsense frame
%   cubeLenL: length of cube edge between lat and lon centres and edge of
%   cube
%   Latitude0S: Case study centre lat
%   Longitude0S: Case study centre lon
% OUTPUT:
%   out: structure that contains 
%       out.lon2            Longitude positions
%       out.lat2            Latitude positions
%       out.outcd           Satsense data (Raw)
%       out.outcdTSmooth:   Satsense data (TSmooth)
%       out.outcdAPS:       Satsense data (APS)
%       out.outcdFilt:      Satsense data (Filt)
%       out.interpDates:    Interpolation dates

% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill July 2019

dateAll = h5read(frameName,'/Date');
theseDates = datenum(num2str(dateAll),'yyyymmdd');
lenD = length(theseDates);
days = theseDates(end)-theseDates(end-1);
interpDates = theseDates(1):days:theseDates(end); interpDates=interpDates';

lat0 = Latitude0S-cubeLenL; lat1 = Latitude0S+cubeLenL;
lon0 = Longitude0S-cubeLenL; lon1 = Longitude0S+cubeLenL;

lonAll = h5read(frameName,'/Longitude');
latAll = h5read(frameName,'/Latitude');

ind2 = ((latAll>lat0)&(latAll<lat1)&(lonAll>lon0)&(lonAll<lon1));

clear latAll 
clear lonAll 

[a, b] = find(ind2); 
minIndLon = min(a);maxIndLon = max(a);
minIndLat = min(b);maxIndLat = max(b);

startL = [minIndLon minIndLat]; countL = [maxIndLon-minIndLon maxIndLat-minIndLat];
startLC = [startL 1]; countLC = [countL lenD];

%cdOrig = h5read(frameName,'/Cumulative_Displacement',startLC,countLC);
cdTSmooth = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);
%cdAPS = h5read(frameName,'/Cumulative_Displacement_APS',startLC,countLC);
%cdFilt = h5read(frameName,'/Cumulative_Displacement_Filt',startLC,countLC);
lon1 = h5read(frameName,'/Longitude', startLC(1:2), countLC(1:2));
lat1 = h5read(frameName,'/Latitude', startLC(1:2), countLC(1:2));
cd1_100 = cdTSmooth(:,:,100);
thisInd2 = isnan(cd1_100);

lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);

%cdOrig = flattenData(cdOrig, lenD, ~thisInd2);
cdTSmooth = flattenData(cdTSmooth, lenD, ~thisInd2);
%cdAPS = flattenData(cdAPS,lenD, ~thisInd2);
%cdFilt = flattenData(cdFilt, lenD, ~thisInd2);
outcdTSmooth = [];
cluster = parcluster('local'); nworkers = cluster.NumWorkers;
parfor (ii = 1:length(cdTSmooth),nworkers)
%for ii = 1:length(cdTSmooth)
        %thisTScd = cdOrig(ii,:); thisTScd = thisTScd(:);
        thisTScdTSmooth = cdTSmooth(ii,:); thisTScdTSmooth = thisTScdTSmooth(:);
        %thisTScdAPS = cdAPS(ii,:); thisTScdAPS = thisTScdAPS(:);
        %thisTScdFilt = cdFilt(ii,:); thisTScdFilt = thisTScdFilt(:);
        %outcd(ii,:) = interp1(theseDates,thisTScd, interpDates);
        outcdTSmooth(ii,:) = interp1(theseDates,thisTScdTSmooth, interpDates);
        %outcdAPS(ii,:) = interp1(theseDates,thisTScdAPS, interpDates);
        %outcdFilt(ii,:) = interp1(theseDates,thisTScdFilt, interpDates);
end

out.lon2 = lon2;
out.lat2 = lat2;
out.outcd = [];
out.outcdTSmooth =outcdTSmooth;
out.outcdAPS = [];
out.outcdFilt = [];
out.interpDates = interpDates;
%out.outcd = outcd;
%out.outcdAPS =outcdAPS;
%out.outcdFilt =outcdFilt;


function cdOrig2 = flattenData(cdOrig, lenD, nNaNInd)
for ii = 1:lenD
    cdOrigN = cdOrig(:,:,ii);
    cdOrigN = cdOrigN(nNaNInd);
    cdOrig2(:,ii) = cdOrigN(:);
end



