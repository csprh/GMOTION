function displaySeaonalData
%% Function that 
%% 1. loads data from interpLocation
%% 2. visualise selected data
%
% USAGE:
%   displaySeaonalData
% INPUT:
%   -
% OUTPUT:
%   -
% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill July 2019
clear; close all;

load interpLocation

signal1D_0= interpLocation.outcd;
signal1D_1= interpLocation.outcdFilt;
signal1D_2= interpLocation.outcdAPS;
signal1D_3= interpLocation.outcdTSmooth;

thisInd = interpLocation.arrayACInd7;

this_signal1DInd0 = signal1D_0(thisInd,:);
this_signal1DInd1 = signal1D_1(thisInd,:);
this_signal1DInd2 = signal1D_2(thisInd,:);
this_signal1DInd3 = signal1D_3(thisInd,:);


thresh1 = 0.0633; thresh2 = 0.1333; thresh3 = 0.2356;
arrayAC = interpLocation.arrayAC;

ind1 = arrayAC<thresh1;
ind2 = arrayAC<thresh2&arrayAC>thresh1;
ind3 = arrayAC<thresh3&arrayAC>thresh2;
ind4 = arrayAC>thresh3;

indind4 = 1:length(ind4);
indi4 = indind4(ind4);


[ACS, ACSInd] = sort(arrayAC);

lat_1 = interpLocation.lat2(ind1); 
lon_1 = interpLocation.lon2(ind1);
lat_2 = interpLocation.lat2(ind2); 
lon_2 = interpLocation.lon2(ind2);
lat_3 = interpLocation.lat2(ind3); 
lon_3 = interpLocation.lon2(ind3);
lat_4 = interpLocation.lat2(ind4); 
lon_4 = interpLocation.lon2(ind4);



lat_1a = interpLocation.lat2(ACSInd(end-10)); 
lon_1a = interpLocation.lon2(ACSInd(end-10));
lat_2a = interpLocation.lat2(ACSInd(end-35)); 
lon_2a = interpLocation.lon2(ACSInd(end-35));
lat_3a = interpLocation.lat2(ACSInd(1)); 
lon_3a = interpLocation.lon2(ACSInd(1));
lat_4a = interpLocation.lat2(ACSInd(25)); 
lon_4a = interpLocation.lon2(ACSInd(25));





hold on;

%A = this_signal1DInd3;
%AI = m2xdate(interpLocation.interpDates);
%Aout = [AI A' ];
%lat2_0 = [0 lat2'];
%lon2_0 = [0 lon2'];
%Aout = [lat2_0; lon2_0; Aout];

%csvwrite('TSmoothSeasonal.csv',Aout);

plot(lon_1,lat_1,'.y','MarkerSize',10);
plot(lon_2,lat_2,'.g','MarkerSize',10);
plot(lon_3,lat_3,'.r','MarkerSize',10);
plot(lon_4,lat_4,'.b','MarkerSize',10);
hold on;

legend('Low Seasonality', 'Mid-Low Seasonality', 'Mid-High Seasonality', 'High Seasonality');
%plot(lon_1a,lat_1a,'+w','MarkerSize',20);
%plot(lon_2a,lat_2a,'+w','MarkerSize',20);
%plot(lon_3a,lat_3a,'+w','MarkerSize',20);
%plot(lon_4a,lat_4a,'+w','MarkerSize',20);


addpath('..');
%plot_google_map('MapScale', 1, 'APIKey','AIzaSyCwSe-kkMTqCkG7jXXhIEgpLv8F5xAZi7U', 'Maptype', 'hybrid');
%plot_google_map('MapScale', 2, 'ShowLabels', 0, 'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
plot_google_map('MapScale', 1,  'ShowLabels', 0,'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');

figure

%subplot(5,1,1);plot(interpLocation.interpDates,this_signal1DInd3(1,:));axis tight; datetick('x','mmm,yy','keepticks');
%subplot(5,1,2);plot(interpLocation.interpDates,this_signal1DInd3(2,:));axis tight; datetick('x','mmm,yy','keepticks');
%subplot(5,1,3);plot(interpLocation.interpDates,this_signal1DInd3(3,:));axis tight; datetick('x','mmm,yy','keepticks');
%subplot(5,1,4);plot(interpLocation.interpDates,this_signal1DInd3(4,:));axis tight; datetick('x','mmm,yy','keepticks');
%subplot(5,1,5);plot(interpLocation.interpDates,this_signal1DInd3(5,:));axis tight; datetick('x','mmm,yy','keepticks');


subplot(4,1,1);plot(interpLocation.interpDates,signal1D_3(ACSInd(end-10),:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,2);plot(interpLocation.interpDates,signal1D_3(ACSInd(end-35),:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,3);plot(interpLocation.interpDates,signal1D_3(ACSInd(1),:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,4);plot(interpLocation.interpDates,signal1D_3(ACSInd(25),:));axis tight; datetick('x','mmm,yy','keepticks');
%subplot(5,1,5);plot(interpLocation.interpDates,signal1D_3(ACSInd(end),:));axis tight; datetick('x','mmm,yy','keepticks');


figure
subplot(4,1,1);plot(interpLocation.interpDates,this_signal1DInd0(2,:)); title('Orig');axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,2);plot(interpLocation.interpDates,this_signal1DInd1(2,:)); title('FILT');axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,3);plot(interpLocation.interpDates,this_signal1DInd2(2,:)); title('APS');axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,4);plot(interpLocation.interpDates,this_signal1DInd3(2,:)); title('TSmooth');axis tight; datetick('x','mmm,yy','keepticks');

