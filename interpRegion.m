function out = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S)

dateAll = h5read(frameName,'/Date');
theseDates = datenum(num2str(dateAll),'yyyymmdd');
days = theseDates(end)-theseDates(end-1);
interpDates = theseDates(1):days:theseDates(end); interpDates=interpDates';

startL = [1 1]; countL = [68941 18482];

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
startLC = [startL 1]; countLC = [countL 171];
lonThis = h5read(frameName,'/Longitude', startL, countL);
latThis = h5read(frameName,'/Latitude', startL, countL);


cdOrig = h5read(frameName,'/Cumulative_Displacement',startLC,countLC);
cdTSmooth = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);
cdAPS = h5read(frameName,'/Cumulative_Displacement_APS',startLC,countLC);
cdFilt = h5read(frameName,'/Cumulative_Displacement_Filt',startLC,countLC);
lon1 = h5read(frameName,'/Longitude', startLC(1:2), countLC(1:2));
lat1 = h5read(frameName,'/Latitude', startLC(1:2), countLC(1:2));
cd1_100 = cdOrig(:,:,100);
thisInd2 = isnan(cd1_100);

cdOrig = cdOrig(~thisInd2);
cdTSmooth = cdTSmooth(~thisInd2);
cdAPS = cdAPS(~thisInd2);
cdFilt = cdFilt(~thisInd2);
lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);


for ii = 1:length(cdOrig)
        thisTScd = cdOrig(ii,:); thisTScd = thisTScd(:);
        thisTScdTSmooth = cdTSmooth(ii,:); thisTScdTSmooth = thisTScdTSmooth(:);
        thisTScdAPS = cdAPS(ii,:); thisTScdAPS = thisTScdAPS(:);
        thisTScdFilt = cdFilt(ii,:); thisTScdFilt = thisTScdFilt(:);
        outcd(ii,:) = interp1(theseDates,thisTScd, interpDates);
        outcdTSmooth(ii,:) = interp1(theseDates,thisTScdTSmooth, interpDates);
        outcdAPS(ii,:) = interp1(theseDates,thisTScdAPS, interpDates);
        outcdFilt(ii,:) = interp1(theseDates,thisTScdFilt, interpDates);
    end
end

out.lon2 = lon2;
out.lat2 = lat2;
out.outcd = cd_1D;
out.cdTSmooth_1D =outcdTSmooth;
out.cdAPS_1D =outcdAPS;
out.cdFilt_1D =outcdFilt;



