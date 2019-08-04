load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';



daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);
%dateAll = h5read(frameName,'/Date');

threshAC0 = 0.2;
threshAC1 = 0.15;
threshAC2 = 0.1;
threshAC3 = 0.05;
load stokeData_cdts2

%signal1D = cd_1D;
%signal1D = cdTSmooth_1D;
signal1D = cdAPS_1D;
%signal1D = cdFilt_1D;

for ii = 1:size(cd_1D,1)
    if rem(ii,10)==0
        ii
    end
    this_signal1D = signal1D(ii,:);

    

    ac = autocorr(this_signal1D,lagAC);
    arrayAC(ii) =  abs(ac(lagAC));
    %> threshAC;
end

arrayACInd0 = arrayAC>threshAC0;
arrayACInd1 = arrayAC>threshAC1;
arrayACInd2 = arrayAC>threshAC2;
arrayACInd3 = arrayAC>threshAC3;

this_signal1DInd0 = signal1D(arrayACInd0,:);
this_signal1DInd1 = signal1D(arrayACInd1,:);
this_signal1DInd2 = signal1D(arrayACInd2,:);
this_signal1DInd3 = signal1D(arrayACInd3,:);

save arrayAC arrayAC this_signal1DInd0 this_signal1DInd1 this_signal1DInd2 this_signal1DInd3 

