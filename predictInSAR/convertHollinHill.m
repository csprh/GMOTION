function convertHollinHill
clear all; close all;
csvName{1} = 'Hollin_hill_081D_03666_031013.csv';
csvName{2} = 'Hollin_hill_154D_03567_081313.csv';
csvName{3} = 'Hollin_hill_132A_03624_131311.csv';

for ii = 1:3
    convThisCSV(csvName{ii});
end

function convThisCSV(thisCSVName)

matName = thisCSVName(1:end-4);
M = importdata(thisCSVName);
cdData = M.data(:,5:end);
labs = M.textdata(1,6:end);

for ii = 1:length(labs)
    thisLab = labs{ii};
    theseDates(ii) = datenum(thisLab(2:end),'yyyymmdd');
end

days = 6
interpDates = theseDates(1):days:theseDates(end); interpDates=interpDates';
for ii = 1:size(cdData,1)
    thisCDData = cdData(ii,:); thisCDData = thisCDData(:);
    outCDData(ii,:) = interp1(theseDates, thisCDData, interpDates);
end
interpLocation.outcdTSmooth = outCDData;
save(matName, 'interpLocation', 'interpDates');%
figure
plot(interpDates, outCDData, '+r'); hold on;
plot(theseDates, cdData, '+'); hold on;


