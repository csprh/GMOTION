load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';



daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);
%dateAll = h5read(frameName,'/Date');

threshAC = 0.5;
load stokeData_cdts2

signal1D = cd_1D;
%singal1D = cdTSmooth_1D;
%singal1D = cdAPS_1D;
%singal1D = cdFilt_1D;

for ii = 1:size(cd_1D,1)
    if rem(ii,10)==0
        ii
    end
    this_singal1D = singal1D(ii,:);

    

    ac = autocorr(this_singal1D,lagAC);
    arrayAC(ii) =  abs(ac(lagAC));
    %> threshAC;
end

arrayACInd = arrayAC>threshAC;

this_signal1DInd = signal1D(arrayACInd,:);
save arrayAC arrayAC this_signal1DInd 

