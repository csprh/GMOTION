function illustrateInterp1


clear all; close all;


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
    thisInterp1 = thisInterpOld(indIn==0);
    thisInterp2 = thisInterpOld(indIn==1);
    indDates2 = interpDates(indIn==1);
    %plot(interpDates, thisInterpOld, 'dr');hold on;
    plot(theseDates, thisInterp1, 'o');hold on;
    plot(indDates2, thisInterp2, '*r');
    %plot(theseDates, zeros(size(thisInterp1,1),1), 'o');
    %plot(indDates2, zeros(size(thisInterp2,1),1), '*r');
    datetick('x', 'yyyy-mm-dd');
    legend('Original Data', 'Interpolated Data');
    xlabel('Date');
    ylabel('InSAR derived displacement');
    ax = gca;
    ax.FontSize = 13; 
end
