function getSeasonalData3

clear; close all;

frameName = '030A_03647_101313-vel.h5'; %Satsense data
loadOut =1;
choosePos = 1; %Normanton coal fields
gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [53.785766, -2.953747]; %New preston road fracking site
gps2 = [54.136083, -1.523892]; %Ripon
gps3 = [53.777 -3.035]; %BLAP
gps4 = [53.800 -1.664]; %LEED

gps = [gps0 ;gps1 ;gps2 ;gps3; gps4];

cubeLenL = 0.01;

thisGPS = gps(choosePos,:);
Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);


if loadOut == 0
    out = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S);
    save out out;
else
    load out;
end

signal1D_0= out.outcd;
signal1D_1= out.outcdFilt;
signal1D_2= out.outcdAPS;
signal1D_3= out.outcdTSmooth;

lat2 = out.lat2; lon2 = out.lon2;
plot(lon2,lat2,'.','color',[0.6,1,0.6]);

daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);

numStds = 7;
for ii = 1:size(signal1D_3,1)
    if rem(ii,100)==0
        ii
    end
    this_signal1D = signal1D_3(ii,:);

    opol = 3;
    t = 1:length(this_signal1D);
    [p,s,mu] = polyfit(t,this_signal1D,opol);
    f_y = polyval(p,t,[],mu);

    this_signal1D = this_signal1D - f_y;
    fullLength = length(this_signal1D);
    thisRem = rem(fullLength, lagAC);
    this_signal1D = this_signal1D(thisRem:end); 

    [acf,lags,bounds] = autocorr(this_signal1D,lagAC,[],numStds);
    arrayAC(ii) =  abs(acf(lagAC+1));
end

arrayACInd = arrayAC>bounds(1);
this_signal1DInd0 = signal1D_0(arrayACInd,:);
this_signal1DInd1 = signal1D_1(arrayACInd,:);
this_signal1DInd2 = signal1D_2(arrayACInd,:);
this_signal1DInd3 = signal1D_3(arrayACInd,:);
hold on;

lat2 = lat2(arrayACInd); lon2 = lon2(arrayACInd);
plot(lon2,lat2,'.r','MarkerSize',20);
addpath('..');
plot_google_map('MapScale', 1, 'APIKey','AIzaSyCwSe-kkMTqCkG7jXXhIEgpLv8F5xAZi7U');

figure
subplot(2,3,1);plot(this_signal1DInd3(1,:)); axis tight
subplot(2,3,2);plot(this_signal1DInd3(2,:));axis tight
subplot(2,3,3);plot(this_signal1DInd3(3,:));axis tight
subplot(2,3,4);plot(this_signal1DInd3(4,:));axis tight
subplot(2,3,5);plot(this_signal1DInd3(5,:));axis tight
subplot(2,3,6);plot(this_signal1DInd3(6,:));axis tight

figure
subplot(4,1,1);plot(this_signal1DInd0(2,:)); title('Orig');axis tight
subplot(4,1,2);plot(this_signal1DInd1(2,:)); title('FILT');axis tight
subplot(4,1,3);plot(this_signal1DInd2(2,:)); title('APS');axis tight
subplot(4,1,4);plot(this_signal1DInd3(2,:)); title('TSmooth');axis tight






