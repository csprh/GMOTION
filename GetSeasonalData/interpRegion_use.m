function out = interpRegion_use(frameName, cubeLenLX, cubeLenLY, Latitude0S, Longitude0S)

dateAll = h5read(frameName,'/Date');
theseDates = datenum(num2str(dateAll),'yyyymmdd');
days = theseDates(end)-theseDates(end-1);
interpDates = theseDates(1):days:theseDates(end); interpDates=interpDates';

startL = [1 1]; countL = [68941 18482];

lat0 = Latitude0S-cubeLenLY; lat1 = Latitude0S+cubeLenLY;
lon0 = Longitude0S-cubeLenLX; lon1 = Longitude0S+cubeLenLX;

lonAll = h5read(frameName,'/Longitude');
latAll = h5read(frameName,'/Latitude');

ind2 = ((latAll>lat0)&(latAll<lat1)&(lonAll>lon0)&(lonAll<lon1));

clear latAll 
clear lonAll 

[a, b] = find(ind2); 
minIndLon = min(a);maxIndLon = max(a);
minIndLat = min(b);maxIndLat = max(b);

startL = [minIndLon minIndLat]; countL = [maxIndLon-minIndLon maxIndLat-minIndLat];
startLC = [startL 1]; countLC = [countL 171];


cdTSmooth = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);
lon1 = h5read(frameName,'/Longitude', startLC(1:2), countLC(1:2));
lat1 = h5read(frameName,'/Latitude', startLC(1:2), countLC(1:2));
cd1_100 = cdTSmooth(:,:,100);
thisInd2 = isnan(cd1_100);


cdTSmooth = cdTSmooth(~thisInd2);

lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);


for ii = 1:length(cdTSmooth) 
        thisTScdTSmooth = cdTSmooth(ii,:); thisTScdTSmooth = thisTScdTSmooth(:);
        outcdTSmooth(ii,:) = interp1(theseDates,thisTScdTSmooth, interpDates);
end

out.lon2 = lon2;
out.lat2 = lat2;

out.cdTSmooth_1D =outcdTSmooth;




