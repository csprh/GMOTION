function getAllEQCubesv1

clear; close all;


latLims =  [52.1914  54.8680];   
lonLims =  [-5.3161  -0.7730];

gpsPos = [53.251 -3.480
;53.777 -3.035
;53.318 -4.642
;52.879 -1.078
;54.840 -4.047
;53.800 -1.664
;53.250 -0.520
;54.563 -0.863
;55.009 -1.433
;54.867 -4.712
;53.736 -0.505
;54.749 -2.230
;54.162 -1.100
]


labs = {'Normaton','Hollin Hill','New preston road','Ripon'};  


gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [54.110862, -0.960384]; %Hollin hill
gps2 = [53.785766, -2.953747]; %New preston road fracking site
gps3 = [54.136083, -1.523892]; %Ripon

gpsPos2 = [gps0;gps1;gps2;gps3];

load latLons

deltaLat = 0.0075;
deltaLon = 0.0125;
ii =1;
  thisLat= gpsPos2(ii,1);
  thisLon= gpsPos2(ii,2);

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

  plot_google_map('MapScale', 1, 'ShowLabels', 0, 'MapType', 'satellite', 'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0cY');
  title(labs{ii});




