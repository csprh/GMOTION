function inFrameQ2

clear; close all;

latLims =  [52.1914  54.8680];   
lonLims =  [-5.3161  -0.7730];

gps0 = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)
gps1 = [53.785766, -2.953747]; %New preston road fracking site
gps2 = [54.136083, -1.523892]; %Ripon
gps3 = [53.777 -3.035]; %BLAP
gps4 = [53.800 -1.664]; %LEED

gps = [gps0 ;gps1 ;gps2 ;gps3; gps4];

cubeLenL = 0.01;

thisGPS = gps(1,:);
Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);

frameName = '030A_03647_101313-vel.h5';
loadOut =1;
if loadOut == 0
    out = interpRegion(frameName, cubeLenL, Latitude0S, Longitude0S);
    save out out;
else
    load out;
end

signal1D= out.outcdAPS;
signal1D_0= out.outcdFilt;
signal1D_1= out.outcdTSmooth;
signal1D_2= out.outcd;

lat2 = out.lat2;
lon2 = out.lon2;
plot(lon2,lat2,'.','color',[0.6,1,0.6]);
daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);

threshAC0 = 0.45;
threshAC1 = 0.5;
for ii = 1:size(signal1D,1)
    if rem(ii,10)==0
        ii
    end
    this_signal1D = signal1D(ii,:);

    opol = 3;
    t = 1:length(this_signal1D);
    [p,s,mu] = polyfit(t,this_signal1D,opol);
    f_y = polyval(p,t,[],mu);

    this_signal1D = this_signal1D - f_y;
    this_signal1D = this_signal1D(40:end); 
    ac = autocorr(smooth(this_signal1D,10),lagAC);
    arrayAC(ii) =  abs(ac(lagAC));

end

arrayACInd0 = arrayAC>threshAC0;
arrayACInd1 = arrayAC>threshAC1;
this_signal1DInd1 = signal1D(arrayACInd1,:);
hold on;

lat2 = lat2(arrayACInd0);
lon2 = lon2(arrayACInd0);
plot(lon2,lat2,'.r','MarkerSize',20);
addpath('..');
plot_google_map('MapScale', 1, 'APIKey','AIzaSyCwSe-kkMTqCkG7jXXhIEgpLv8F5xAZi7U');

figure
subplot(2,3,1);plot(this_signal1DInd1(1,:)); axis tight
subplot(2,3,2);plot(this_signal1DInd1(2,:));axis tight
subplot(2,3,3);plot(this_signal1DInd1(3,:));axis tight
subplot(2,3,4);plot(this_signal1DInd1(4,:));axis tight
subplot(2,3,5);plot(this_signal1DInd1(5,:));axis tight
subplot(2,3,6);plot(this_signal1DInd1(6,:));axis tight





