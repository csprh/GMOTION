function getSeasonalData_Use
%% Function that
%% 1. extracts data from satsense frame
%% 2. finds the data that surrounds a datapoint of interest
%% 3. interpolates missing data
%% 4. get's indices for the data indentified as seasonal (5, 6 and 7 sigma)
%
% USAGE:
%   getSeasonalData
% INPUT:
%   -
% OUTPUT:
%   save output to matlab file
% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill July 2019

clear; close all;

frameName = '../030A_03647_101313-vel.h5'; %Satsense data
loadInterpLocation = 1;
thisGPS  = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)


cubeLenLX = 0.0125;
cubeLenLY = 0.0075;

Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);
load theseDates;

if loadInterpLocation == 0
    interpLocation = interpRegion_use(frameName,cubeLenLX, cubeLenLY, Latitude0S, Longitude0S);
    save interpLocation interpLocation;
else
    load interpLocation;
end

signal1D= interpLocation.outcdTSmooth;

lat2 = interpLocation.lat2; lon2 = interpLocation.lon2;
loadUse = 1;
if loadUse ~= 1;
    
    limsLon = [min(lon2) max(lon2)];
    limsLat = [min(lat2) max(lat2)];
    
    divsLon = 20;
    divsLat = 20;
    
    thisUseLong = zeros(size(lat2));
    for iiLon = 1: divsLon
        for iiLat = 1: divsLat
            
            thisLimsLon(1) = limsLon(1) + (iiLon-1)* (limsLon(2)-limsLon(1))/divsLon;
            thisLimsLon(2) = limsLon(1) + (iiLon)* (limsLon(2)-limsLon(1))/divsLon;
            thisLimsLat(1) = limsLat(1) + (iiLat-1)* (limsLat(2)-limsLat(1))/divsLat;
            thisLimsLat(2) = limsLat(1) + (iiLat)* (limsLat(2)-limsLat(1))/divsLat;
            
            
            plot(thisLimsLon(1),thisLimsLat(1),'.b'); hold on;
            plot(thisLimsLon(1),thisLimsLat(2),'.b'); hold on;
            plot(thisLimsLon(2),thisLimsLat(1),'.b'); hold on;
            plot(thisLimsLon(2),thisLimsLat(2),'.b'); hold on;
            theseInds = (lon2>thisLimsLon(1))&(lon2<thisLimsLon(2))&(lat2>thisLimsLat(1))&(lat2<thisLimsLat(2));
            
            if sum(theseInds) ~=0
                longInds = 1:length(theseInds);
                actualInds = longInds(theseInds');
                theseLats = lat2(theseInds);
                theseLons = lon2(theseInds);
                thisFig1 = figure;
                plot(thisLimsLon(1),thisLimsLat(1),'+g'); hold on;
                plot(thisLimsLon(1),thisLimsLat(2),'+g'); hold on;
                plot(thisLimsLon(2),thisLimsLat(1),'+g'); hold on;
                plot(thisLimsLon(2),thisLimsLat(2),'+g'); hold on;
                plot(theseLons,theseLats,'+b');
                styleStr = 'feature:road%7Ccolor:0xff0000&style=feature:water%7Ccolor:0x0000ff&style=feature:transit.line%7Ccolor:0x00ffff';
                gMap = plot_google_map('ShowLabels', 0, 'style', styleStr, 'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
                
                buildInd = getBuildings(gMap);
                
                thisFig2 = figure;
                plot(thisLimsLon(1),thisLimsLat(1),'.b'); hold on;
                plot(thisLimsLon(1),thisLimsLat(2),'.b'); hold on;
                plot(thisLimsLon(2),thisLimsLat(1),'.b'); hold on;
                plot(thisLimsLon(2),thisLimsLat(2),'.b'); hold on;
                plot(theseLons,theseLats,'.b'); hold on;
                styleStr = 'feature:road%7Ccolor:0xff0000&style=feature:landscape.man_made%7Ccolor:0x00ff00&style=feature:water%7Ccolor:0x0000ff&style=feature:transit.line%7Ccolor:0x00ffff';
                gMap = plot_google_map('ShowLabels', 0, 'style', styleStr, 'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
                roadsInd = getRoads(gMap);
                manInd= getManMade(gMap);
                railInd = getRail(gMap);
                X = gMap.XData;
                Y = gMap.YData;
                for ii = 1: length(theseLats)
                    [~, XInd] = min(abs(theseLons(ii)-X));
                    [~, YInd] = min(abs(theseLats(ii)-Y));
                    isBuild = buildInd(YInd,XInd);
                    isRoads = roadsInd(YInd,XInd);
                    isMan = manInd(YInd,XInd);
                    isRail = railInd(YInd,XInd);
                    thisUse(ii) = isBuild + isRoads*2 + isMan *4 + isRail*8;
                    thisUseLong(actualInds(ii)) = thisUse(ii);
                end
                
                close all;
                %figure;
                %subplot(2,3,1);            imagesc(buildInd);
                %subplot(2,3,2);            imagesc(roadsInd);
                %subplot(2,3,3);            imagesc(manInd);
                %subplot(2,3,4);            imagesc(railInd);
                %subplot(2,3,5);            imagesc(thisIm);
                %pause;
                
            end
        end
    end
    
    save usageHere thisUseLong lat2 lon2
    
else
    load usageHere
end
figure;
buildLats = lat2(bitand(thisUseLong,1)~=0);
buildLons = lon2(bitand(thisUseLong,1)~=0);

%plot(buildLons,buildLats,'.b'); hold on;

roadsLats = lat2(bitand(thisUseLong,2)~=0);
roadsLons = lon2(bitand(thisUseLong,2)~=0);

plot(roadsLons,roadsLats,'.r'); hold on;

manLats = lat2(bitand(thisUseLong,4)~=0);
manLons = lon2(bitand(thisUseLong,4)~=0);

plot(manLons,manLats,'.g'); hold on;

railLats = lat2(bitand(thisUseLong,8)~=0);
railLons = lon2(bitand(thisUseLong,8)~=0);

plot(railLons,railLats,'.k'); hold on;

noneLats = lat2(thisUseLong==0);
noneLons = lon2(thisUseLong==0);

plot(noneLons,noneLats,'.b'); hold on;
%plot_google_map('MapScale', 1, 'ShowLabels', 0, 'MapType', 'satellite', 'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
gMap = plot_google_map('Scale', 2, 'ShowLabels', 0,'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');
X = gMap.XData;
Y = gMap.YData;
outIm = gMap.CData;
[h,icons] = legend('Roads', 'Man Made', 'Rail', 'Other');


icons = findobj(icons,'Type','line');
icons = findobj(icons,'Marker','none','-xor');
set(icons,'MarkerSize',20);
h.FontSize = 18;

soilIm = imread('soil.png');figure;imagesc(soilIm);figure

mineIm = imread('mines.png');figure;imagesc(mineIm);figure

mineInd = (mineIm~=0);
mineInd = mineInd(:,:,1);

soil1Ind = getSoil1(soilIm); %imagesc(soil1Ind); figure
soil2Ind = getSoil2(soilIm); %imagesc(soil2Ind); figure
soil3Ind = getSoil3(soilIm); %imagesc(soil3Ind); figure
soil4Ind = getSoil4(soilIm); %imagesc(soil4Ind);

outInd = soil1Ind*1+soil2Ind*10+soil3Ind*30+soil4Ind*40;
figure; imagesc(outInd);


for ii = 1: length(lat2)
    [~, XInd] = min(abs(lon2(ii)-X));
    [~, YInd] = min(abs(lat2(ii)-Y));
    outIm(YInd,XInd,:) = 0;
    isSoil1 = soil1Ind(YInd,XInd);
    isSoil2 = soil2Ind(YInd,XInd);
    isSoil3 = soil3Ind(YInd,XInd);
    isSoil4 = soil4Ind(YInd,XInd);

    isMine = mineInd(YInd,XInd);
    thisUseSoil(ii) = isSoil1 + isSoil2*2 + isSoil3 *4 + isSoil4*8 + isMine*16; 
end

figure;
mineLats = lat2(bitand(thisUseSoil,16)~=0);
mineLons = lon2(bitand(thisUseSoil,16)~=0);

nomineLats = lat2(bitand(thisUseSoil,16)==0);
nomineLons = lon2(bitand(thisUseSoil,16)==0);
plot(mineLons,mineLats,'.b'); hold on;
plot(nomineLons,nomineLats,'.r'); hold on;

gMap = plot_google_map('Scale', 2, 'ShowLabels', 0,'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');

[h,icons] = legend('Shallow Mining', 'Non Shallow Mining');

icons = findobj(icons,'Type','line');
icons = findobj(icons,'Marker','none','-xor');
set(icons,'MarkerSize',20);
h.FontSize = 18;


figure;


soil1Lats = lat2(bitand(thisUseSoil,1)~=0);
soil1Lons = lon2(bitand(thisUseSoil,1)~=0);

plot(soil1Lons,soil1Lats,'.b'); hold on;

soil2Lats = lat2(bitand(thisUseSoil,2)~=0);
soil2Lons = lon2(bitand(thisUseSoil,2)~=0);

plot(soil2Lons,soil2Lats,'.r'); hold on;

soil3Lats = lat2(bitand(thisUseSoil,4)~=0);
soil3Lons = lon2(bitand(thisUseSoil,4)~=0);

plot(soil3Lons,soil3Lats,'.g'); hold on;

soil4Lats = lat2(bitand(thisUseSoil,8)~=0);
soil4Lons = lon2(bitand(thisUseSoil,8)~=0);

plot(soil4Lons,soil4Lats,'.k'); hold on;

gMap = plot_google_map('Scale', 2, 'ShowLabels', 0,'APIKey','AIzaSyA4GhtMt4rP_0YZa65CV1A1VZzRBV_0c_Y');

[h,icons] = legend('Intermediate-Shallow, Clayey Loam to Sandy Loam, Mudstone and Sandstone','Shallow, Loam to Sandy Loam, Sandstone','Deep, Riverine clay: Sands and Gravels, Clay to Sandy Loam','Deep, River Terrace: Sand gravel, Sandy Loam');

icons = findobj(icons,'Type','line');
icons = findobj(icons,'Marker','none','-xor');
set(icons,'MarkerSize',20);
h.FontSize = 18;


daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);

soil0OutInd = thisUseSoil==0;
soil1OutInd = bitand(thisUseSoil,1)~=0;
soil2OutInd = bitand(thisUseSoil,2)~=0;
soil3OutInd = bitand(thisUseSoil,4)~=0;
soil4OutInd = bitand(thisUseSoil,8)~=0;
mineOutInd = bitand(thisUseSoil,16)~=0;
manIndThis = bitand(thisUseLong,4)~=0;
%manIndThis = isSoil1==1;
%manIndThis = thisUseLong==0;
for ii = 1:size(signal1D,1)
    this_signal1D = signal1D(ii,:);
    meanSig(ii) = mean(this_signal1D);
    % Remove 3rd degree polynomial trend
    opol = 3;
    t = 1:length(this_signal1D);
    [p,s,mu] = polyfit(t,this_signal1D,opol);
    f_y = polyval(p,t,[],mu);
    this_signal1D = this_signal1D - f_y;
    
    % Remove bit of the time series that makes it a full number of years
    fullLength = length(this_signal1D);
    thisRem = rem(fullLength, lagAC);
    this_signal1D = this_signal1D(thisRem+1:end);
    
    freq = abs(fft(this_signal1D));
    % Find autocorrelation
    acf = autocorr(this_signal1D,lagAC);
    
    vcrit = sqrt(2)*erfinv(0.95);
    lconf = -vcrit/sqrt(length(this_signal1D));
    
    0.0633; thresh2 = 0.1333; thresh3 = 0.2356;
    
    N = length(this_signal1D);
    vcrit1 = 0.0633*sqrt(N); conf1 = erf(vcrit1/sqrt(2));
    vcrit2 = 0.1333*sqrt(N); conf2 = erf(vcrit2/sqrt(2));
    vcrit3 = 0.2356*sqrt(N); conf3 = erf(vcrit3/sqrt(2));
    
    [acf2,lags,bounds] = autocorr(this_signal1D,lagAC,[],vcrit);
    arrayAC(ii) =  abs(acf(lagAC+1));
    %arrayAC(ii) =  freq(4);
    
end


meanSigMine = meanSig(mineOutInd);
meanSigNoMine = meanSig(~mineOutInd);

[Manp, Manq ] = hist(meanSigMine,100);
[noManp, noManq ] = hist(meanSigNoMine,100);

figure;
plot(noManq,smooth(noManp/sum(noManp)),'b');
hold on;plot(Manq,smooth(Manp/sum(Manp)),'r');
legend('Null','Shallow Coal Workings');
xlabel('Displacement');
ylabel('Normalised Histogram Freq');

arrayACMan = arrayAC(manIndThis);
arrayACNoMan = arrayAC(~manIndThis);

[Manp, Manq ] = hist(arrayACMan,100);
[noManp, noManq ] = hist(arrayACNoMan,100);

figure;
plot(noManq,smooth(noManp/sum(noManp)),'b');
hold on;plot(Manq,smooth(Manp/sum(Manp)),'r');
legend('Not Man Made','Man Made');
xlabel('Seasonality Index');
ylabel('Normalised Histogram Freq');

arrayACSoil0 = arrayAC(soil0OutInd);
arrayACSoil1 = arrayAC(soil1OutInd);
arrayACSoil2 = arrayAC(soil2OutInd);
arrayACSoil3 = arrayAC(soil3OutInd);
arrayACSoil4 = arrayAC(soil4OutInd);
arrayACMine = arrayAC(mineOutInd);


%%%%%%%%%%%%%%%%%%%%STATS TEST%%%%%%%%%%%%%%%%%%%
NOTarrayACSoil0=arrayAC(~soil0OutInd);
NOTarrayACSoil1=arrayAC(~soil1OutInd);
NOTarrayACSoil2=arrayAC(~soil2OutInd);
NOTarrayACSoil3=arrayAC(~soil3OutInd);
NOTarrayACSoil4=arrayAC(~soil4OutInd);
NOTarrayACMine = arrayAC(~mineOutInd);

[H0,P0] = ttest2(NOTarrayACSoil0, arrayACSoil0);
[H1,P1] = ttest2(NOTarrayACSoil1, arrayACSoil1);
[H2,P2] = ttest2(NOTarrayACSoil2, arrayACSoil2);
[H3,P3] = ttest2(NOTarrayACSoil3, arrayACSoil3);
[H4,P4] = ttest2(NOTarrayACSoil4, arrayACSoil4);
[H5,PM] = ttest2(NOTarrayACMine,  arrayACMine);

[H0,P0] = kstest2(NOTarrayACSoil0, arrayACSoil0);
[H1,P1] = kstest2(NOTarrayACSoil1, arrayACSoil1);
[H2,P2] = kstest2(NOTarrayACSoil2, arrayACSoil2);
[H3,P3] = kstest2(NOTarrayACSoil3, arrayACSoil3);
[H4,P4] = kstest2(NOTarrayACSoil4, arrayACSoil4);
[HM,PM] = kstest2(NOTarrayACMine,  arrayACMine);


[arrayACSoil0p, arrayACSoil0q ] = hist(arrayACSoil0,50);
[arrayACSoil1p, arrayACSoil1q ] = hist(arrayACSoil1,50);
[arrayACSoil2p, arrayACSoil2q ] = hist(arrayACSoil2,50);
[arrayACSoil3p, arrayACSoil3q ] = hist(arrayACSoil3,50);
[arrayACSoil4p, arrayACSoil4q ] = hist(arrayACSoil4,50);
[arrayACMinep, arrayACMineq ] = hist(arrayACMine,50);

figure;
plot(arrayACSoil0q,smooth(arrayACSoil0p/sum(arrayACSoil0p),10),'b:');hold on;
plot(arrayACSoil1q,smooth(arrayACSoil1p/sum(arrayACSoil1p)),'b');
plot(arrayACSoil2q,smooth(arrayACSoil2p/sum(arrayACSoil2p)),'r');
plot(arrayACSoil3q,smooth(arrayACSoil3p/sum(arrayACSoil3p)),'g');
plot(arrayACSoil4q,smooth(arrayACSoil4p/sum(arrayACSoil4p),20),'k');
plot(arrayACMineq,smooth(arrayACMinep/sum(arrayACMinep),20),'k:');

legend({'Unclassified','Intermediate-Shallow, Clayey Loam to Sandy Loam, Mudstone and Sandstone','Shallow, Loam to Sandy Loam, Sandstone','Deep, Riverine clay: Sands and Gravels, Clay to Sandy Loam','Deep, River Terrace: Sand gravel, Sandy Loam','Shallow Mining'}, 'FontSize',16);
xlabel('Seasonality Index','FontSize',16);
ylabel('Normalised Histogram Freq', 'FontSize',16);
% Obtrain 5,6 and 7 sigma bounds on the autocorrelation
numStds = 5; [~,~,bounds5] = autocorr(this_signal1D,lagAC,[],numStds);
numStds = 6; [~,~,bounds6] = autocorr(this_signal1D,lagAC,[],numStds);
numStds = 7; [~,~,bounds7] = autocorr(this_signal1D,lagAC,[],numStds);

arrayACInd5 = arrayAC>bounds5(1);
arrayACInd6 = arrayAC>bounds6(1);
arrayACInd7 = arrayAC>bounds7(1);

interpLocation.arrayACInd5 = arrayACInd5;
interpLocation.arrayACInd6 = arrayACInd6;
interpLocation.arrayACInd7 = arrayACInd7;
interpLocation.arrayAC = arrayAC;

save interpLocation interpLocation;



function [thisIndOut] = getSoil1(thisIm)

bIndMinMaxR = [180 256];
bIndMinMaxG = [-1 50];
bIndMinMaxB = [-1 50];
thisIndOut = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));


function [thisIndOut] = getSoil2(thisIm)

bIndMinMaxR = [250 256];
bIndMinMaxG = [250 256];
bIndMinMaxB = [-1 30];
thisIndOut = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));

function [thisIndOut] = getSoil3(thisIm)

bIndMinMaxR = [-1 10];
bIndMinMaxG = [243 256];
bIndMinMaxB = [-1 10];
thisIndOut = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));

function [thisIndOut] = getSoil4(thisIm)
bIndMinMaxR = [250 256];
bIndMinMaxG = [250 256];
bIndMinMaxB = [250 256];
thisIndOut = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));



function [thisIndOut] = getManMade(gMap)
thisIm = gMap.CData;
bIndMinMaxR = [-1 1];
bIndMinMaxG = [243 256];
bIndMinMaxB = [-1 1];
thisInd = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));
thisIndOut = medfilt2(thisInd);


function [thisIndOut] = getRoads(gMap)
thisIm = gMap.CData;
bIndMinMaxR = [243 256];
bIndMinMaxG = [-1 1];
bIndMinMaxB = [-1 1];
thisInd = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));
thisIndOut = medfilt2(thisInd);

function [thisIndOut] = getBuildings(gMap)

bIndMinMaxR = [225 242];
bIndMinMaxG = [225 242];
bIndMinMaxB = [225 242];


thisIm = gMap.CData;
thisInd = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));
thisInd1 = medfilt2(thisInd);


bIndMinMaxR = [243 256];
bIndMinMaxG = [243 253];
bIndMinMaxB = [233 244];
thisInd = (thisIm(:,:,1)>bIndMinMaxR(1)) & (thisIm(:,:,1)<bIndMinMaxR(2)) & (thisIm(:,:,2)>bIndMinMaxG(1)) & (thisIm(:,:,2)<bIndMinMaxG(2)) & (thisIm(:,:,3)>bIndMinMaxB(1)) & (thisIm(:,:,3)<bIndMinMaxB(2));
thisInd2 = medfilt2(thisInd);

thisIndOut = thisInd1|thisInd2;

