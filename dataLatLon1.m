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

labs = {'ASAP','BLAP','HOLY','KEYW','KIRK','LEED','LOFT','STRN','WEAR','YEAS'};  


gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [54.110862, -0.960384]; %Normaton coal fields
gps2 = [53.777 -3.035]; %BLAP
gps3 = [53.800 -1.664]; %LEED



for ii = 1: size(gpsPos,1);
   thislat= gpsPos(ii,1);
   thislon= gpsPos(ii,2);
   ind(ii) = false;
   if (thislat>latLims(1))&&(thislat<latLims(2))&&(thislon>lonLims(1))&&(thislon<lonLims(2))
      ind(ii) = true; 
   end
end

gpsPos2 = gpsPos(ind,:);
gpsPos2

load latLons
deltaLat = 0.01;
deltaLon = 0.01;
for ii = 1: size(gpsPos2,1);
  subplot(5,2,ii);
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

  plot_google_map('MapScale', 1, 'APIKey','AIzaSyBQ5Px_1bwuiPH9Nz94jeaWncRtTYcB2m');
  title(labs{ii});
   
end




