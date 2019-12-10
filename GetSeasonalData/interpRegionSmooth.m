function out = interpRegionSmooth(frameName, cubeLenLx, cubeLenLy,Latitude0S, Longitude0S)
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
%       out.outcdTSmooth:   Satsense data (TSmooth)
%       out.interpDates:    Interpolation dates

% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill July 2019

dateAll = h5read(frameName,'/Date');
theseDates = datenum(num2str(dateAll),'yyyymmdd');
lenD = length(theseDates);
days = theseDates(end)-theseDates(end-1);
interpDates = theseDates(1):days:theseDates(end); interpDates=interpDates';

lat0 = Latitude0S-cubeLenLy; lat1 = Latitude0S+cubeLenLy;
lon0 = Longitude0S-cubeLenLx; lon1 = Longitude0S+cubeLenLx;

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

cdTSmooth = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);

lon1 = h5read(frameName,'/Longitude', startLC(1:2), countLC(1:2));
lat1 = h5read(frameName,'/Latitude', startLC(1:2), countLC(1:2));
cd1_100 = cdTSmooth(:,:,100);
thisInd2 = isnan(cd1_100);

lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);


cdTSmooth = flattenData(cdTSmooth, lenD, ~thisInd2);


for ii = 1:length(cdTSmooth)
        thisTScdTSmooth = cdTSmooth(ii,:); thisTScdTSmooth = thisTScdTSmooth(:);
        outcdTSmooth(ii,:) = interp1(theseDates,thisTScdTSmooth, interpDates);
end

out.lon2 = lon2;
out.lat2 = lat2;
out.outcdTSmooth =outcdTSmooth;
out.interpDates = interpDates;

function cdOrig2 = flattenData(cdOrig, lenD, nNaNInd)
for ii = 1:lenD
    cdOrigN = cdOrig(:,:,ii);
    cdOrigN = cdOrigN(nNaNInd);
    cdOrig2(:,ii) = cdOrigN(:);
end



