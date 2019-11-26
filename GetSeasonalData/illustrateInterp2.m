function illustrateInterp2


clear; close all;


load theseDates;

load interpLocation;

interpDates = interpLocation.interpDates;
aAC = interpLocation.arrayAC;
[pp, aa] = sort(aAC);

lengthOfIn = size(interpLocation.outcdTSmooth,2);
indIn = ones(lengthOfIn,1);
for iii = 1: length(theseDates)
    indIn(interpDates==theseDates(iii)) = 0;
end
for ii = 1: 10
    thisInterpOld = interpLocation.outcdTSmooth(aa(end-ii-1),:);
    thisInterp = thisInterpOld(indIn'==1);
    plot(interpDates, thisInterpOld, 'd');hold on;
    plot(theseDates, thisInterp, 'sr');
    
end
