function extractGPS

clear; close all;

gpsStationName = 'LEED_LOS';
gpsStationName = 'KEYW_LOS';
gpsStationName = 'DARE_LOS';
gpsStationName = 'BLAP_LOS';
gpsStationName = 'ASAP_LOS';

A = readtable(gpsStationName);
dn  = datenum(A{:,1});
dis = A{:,2};
dis = smooth(dis,50);
plot(dn,dis);

