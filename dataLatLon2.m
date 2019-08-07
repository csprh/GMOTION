function inFrameQ2

clear; close all;

latLims =  [52.1914  54.8680];   
lonLims =  [-5.3161  -0.7730];

gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [54.110862, -0.960384]; %Hollin Hill
gps2 = [53.777 -3.035]; %BLAP
gps3 = [53.800 -1.664]; %LEED

gps = [gps0 ;gps1 ;gps2 ;gps3];

cubeLenL = 0.01;

thisGPS = gps(1,:);
Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);

frameName = '030A_03647_101313-vel.h5';

startL = [1 1]; countL = [68941 18482];
startLC = [startL 100]; countLC = [countL 100];

lat0 = Latitude0S-cubeLenL; lat1 = Latitude0S+cubeLenL;
lon0 = Longitude0S-cubeLenL; lon1 = Longitude0S+cubeLenL;

lonAll = h5read(frameName,'/Longitude');
latAll = h5read(frameName,'/Latitude');

ind2 = ((latAll>lat0)&(latAll<lat1)&(lonAll>lon0)&(lonAll<lon1));

clear latAll 
clear lonAll 

dateAll = h5read(frameName,'/Date');

[a, b] = find(ind2); 
minIndLon = min(a);maxIndLon = max(a);
minIndLat = min(b);maxIndLat = max(b);

startL = [minIndLon minIndLat]; countL = [maxIndLon-minIndLon maxIndLat-minIndLat];
startLC = [startL 1]; countLC = [countL 170];
lonThis = h5read(frameName,'/Longitude', startL, countL);
latThis = h5read(frameName,'/Latitude', startL, countL);



cThis = h5read(frameName,'/Cumulative_Displacement_TSmooth',startLC,countLC);
cThis_100 = cThis(:,:,100);
thisInd2 = isnan(cThis_100);
ind2 = ((latThis>lat0)&(latThis<lat1)&(lonThis>lon0)&(lonThis<lon1));
lon2 = lonThis((~thisInd2)&ind2);
lat2 = latThis((~thisInd2)&ind2);








load latLons
deltaLat = 0.01;
deltaLon = 0.01;
for ii = 1: size(gpsPos2,1);
  subplot(5,2,ii);
  thisLat= gpsPos(ii,1);
  thisLon= gpsPos(ii,2);

  minLat = thisLat - deltaLat; 
  maxLat = thisLat + deltaLat;
  minLon = thisLon - deltaLon;
  maxLon = thisLon + deltaLon;
  
  ind = (lat2>minLat)&(lat2<maxLat)&(lon2>minLon)&(lon2<maxLon);
  
  lon3 = lon2(ind);
  lat3 = lat2(ind);

  
  plot(lon3, lat3, '.b');
  hold on;
  plot(thisLon, thisLat, '.r', 'MarkerSize', 20);
  plot_google_map('MapScale', 1, 'APIKey','AIzaSyBQ5Px_1bwuiPH9Nz94jeaWncRtTYcB2m8');
  title(labs{ii});
   
end




