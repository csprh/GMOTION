load  UKEQs2016-2018 

worldmap([49 59],[-12 4]) 
geoshow('landareas.shp') 

DScale = 500000;
Latitude0 = Latitude(ML==0); Longitude0 = Longitude(ML==0); thisDate0 = thisDate(ML==0);
Latitude1 = Latitude(ML~=0); Longitude1 = Longitude(ML~=0); thisDate1 = thisDate(ML~=0);

scaleD0 = (thisDate0 - min(thisDate0(:)))./(max(thisDate0(:))-min(thisDate0(:)));
hold on;
view(3)

cubeLenL = 0.15/10;
cubeLenD = 0.025/10;
Longitude0S = Longitude0(44);
Latitude0S = Latitude0(44);
scaleD0S = scaleD0(44); 

lat0 = Latitude0S-cubeLenL; lat1 = Latitude0S+cubeLenL;
lon0 = Longitude0S-cubeLenL; lon1 = Longitude0S+cubeLenL;

lonAll = h5read('030A_03647_101313-vel.h5','/Longitude');

%[~, minIndLon] = min(abs(lonAll-lon0));
%[~, maxIndLon] = min(abs(lonAll-lon1));



%minIndLon = min(minIndLon);
%maxIndLon = max(maxIndLon);
%clear lonAll 

latAll = h5read('030A_03647_101313-vel.h5','/Latitude');

ind2 = ((latAll>lat0)&(latAll<lat1)&(lonAll>lon0)&(lonAll<lon1));

clear latAll 
clear lonAll 
%[~, minIndLat] = min(abs(latAll'-lat0));
%[~, maxIndLat] = min(abs(latAll'-lat1));
%minIndLat = min(minIndLat);
%maxIndLat = max(maxIndLat);
%clear latAll 
dateAll = h5read('030A_03647_101313-vel.h5','/Date');

[a, b] = find(ind2); 
minIndLon = min(a);maxIndLon = max(a);

minIndLat = min(b);maxIndLat = max(b);

startL = [minIndLon minIndLat]; countL = [maxIndLon-minIndLon maxIndLat-minIndLat];
startLC = [startL 1]; countLC = [countL 170];
lonThis = h5read('030A_03647_101313-vel.h5','/Longitude', startL, countL);
latThis = h5read('030A_03647_101313-vel.h5','/Latitude', startL, countL);


%cd1 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
cThis = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement_TSmooth',startLC,countLC);
cThis_100 = cThis(:,:,100);
thisInd2 = isnan(cThis_100);
ind2 = ((latThis>lat0)&(latThis<lat1)&(lonThis>lon0)&(lonThis<lon1));
lon2 = lonThis((~thisInd2)&ind2);
lat2 = latThis((~thisInd2)&ind2);

allTriplets  = [];
for ii = 2:170

	cThisFrame = cThis(:,:,ii);

	thisInd1 = isnan(cThisFrame);
    cThisFrameNoNaN = cThisFrame((~thisInd1)&ind2);
    thisDate = dateAll(ii);
    thisTriplet = [lon2 lat2 (single(thisDate)*ones(size(lon2))) cThisFrameNoNaN];
    allTriplets = [thisTriplet; allTriplets];
end	

save demo44Triplets allTriplets
noOfPoints = size(allTriplets,1);
for ii =1:noOfPoints
end

