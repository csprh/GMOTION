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

lat2 = interpLocation.lat2(thisInd); 
lon2 = interpLocation.lon2(thisInd);
hold on;

A = this_signal1DInd3;
AI = m2xdate(interpLocation.interpDates);
Aout = [AI A' ];
lat2_0 = [0 lat2'];
lon2_0 = [0 lon2'];
Aout = [lat2_0; lon2_0; Aout];

csvwrite('TSmoothSeasonal.csv',Aout);

plot(lon2,lat2,'.r','MarkerSize',20);
addpath('..');
plot_google_map('MapScale', 1, 'APIKey','AIzaSyCwSe-kkMTqCkG7jXXhIEgpLv8F5xAZi7');
%U
figure
subplot(5,1,1);plot(interpLocation.interpDates,this_signal1DInd3(1,:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(5,1,2);plot(interpLocation.interpDates,this_signal1DInd3(2,:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(5,1,3);plot(interpLocation.interpDates,this_signal1DInd3(3,:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(5,1,4);plot(interpLocation.interpDates,this_signal1DInd3(4,:));axis tight; datetick('x','mmm,yy','keepticks');
subplot(5,1,5);plot(interpLocation.interpDates,this_signal1DInd3(5,:));axis tight; datetick('x','mmm,yy','keepticks');


figure
subplot(4,1,1);plot(interpLocation.interpDates,this_signal1DInd0(2,:)); title('Orig');axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,2);plot(interpLocation.interpDates,this_signal1DInd1(2,:)); title('FILT');axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,3);plot(interpLocation.interpDates,this_signal1DInd2(2,:)); title('APS');axis tight; datetick('x','mmm,yy','keepticks');
subplot(4,1,4);plot(interpLocation.interpDates,this_signal1DInd3(2,:)); title('TSmooth');axis tight; datetick('x','mmm,yy','keepticks');

