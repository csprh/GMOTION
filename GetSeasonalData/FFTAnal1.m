function getSeasonalData_AllMetrics_AC
% Choose either All satsense data or Normanton
% Define Name (either All or Norm)
% Output seasonality Index
% 
%
% USAGE:
%   getSeasonalData_AllMetrics
% INPUT:
%   -
% OUTPUT:
%   save output to matlab file
% THE UNIVERSITY OF BRISTOL: Digital Environment

% Author Dr Paul Hill Dec 2019

clear; close all;

loadInterpLocation = 1; normanton = 1; doPlot = 1;

frameName = '../030A_03647_101313-vel.h5'; %Satsense data


thisGPS = [53.706800, -1.391170]; %Normaton coal fields (the West Yorkshire coalfields)

if normanton == 1
    cubeLenLx = 0.01;
    cubeLenLy = 0.01;
    interpLocationName = 'interpLocationNorm';
else
    cubeLenLx = 1000;
    cubeLenLy = 1000;
    interpLocationName = 'interpLocationAll';
end

Latitude0S = thisGPS(1);
Longitude0S = thisGPS(2);


if loadInterpLocation == 0
    interpLocation = interpRegionSmooth(frameName, cubeLenLx, cubeLenLy, Latitude0S, Longitude0S);
    save (interpLocationName, 'interpLocation');
else
    load (interpLocationName);
end

x = interpLocation.outcdTSmooth;
%periodogram(x,hamming(length(x)));

Fs = 61;                                        % Sampling Frequency (Samples / Year)
Ts = 1/Fs;                                      % Sampling Interval (Year/Samples)
Fn = Fs/2;                                      % Nyquist Frequency (0.5*Samples / Year)
L  = 222;                                       % Signal Length (samples)
t = [0:(L-1)]*Ts;                               % Time Vector (1/Days)
s = cos(2*pi*t);                                % Signal ()


xx = 1: 222

pp = sin(2*pi * (xx / (61.0)));

[pxx2,w2] = periodogram(pp,hamming(size(x,2)),1024);
[pxx,w] = periodogram(x',hamming(size(x,2)),1024);


pxx = 10*log10(pxx');

samplingFreq = 61;
eachBin = samplingFreq / 257;

pxx = pxx(:,1:64);
[histOut,X] = hist(pxx,400)
imagesc((flipud(log(histOut))));
xlabel('Normalised Frequency (Cycles Per Year)', 'fontsize', 14);
xticklabels = [0 0.0625 0.125];
xticks = linspace(1, size(histOut, 2), numel(xticklabels));
xticks = 0:18:5*18;
set(gca, 'XTick', xticks, 'XTickLabel', xticks/18)
%set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

ylabel('PSD (dB)', 'fontsize', 14);
yticklabels = X(400:-40:1);
yticks = linspace(1, size(histOut, 1), numel(yticklabels));

set(gca, 'YTick', yticks, 'YTickLabel', yticklabels);

tix=get(gca,'ytick')';
line([18,18],[405, 1], 'LineStyle', '--', 'Color', 'yellow');
line([36,36],[405, 1], 'LineStyle', '--', 'Color', 'yellow');
line([56,56],[405, 1], 'LineStyle', '--', 'Color', 'yellow');
set(gca,'yticklabel',num2str(yticklabels,'%.2f'))

%plot(w,10*log10(pxx))