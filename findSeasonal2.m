load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';



daysBetweenSamples = 6;
daysInYear = 365.25;
lagAC = round(daysInYear/daysBetweenSamples);
%dateAll = h5read(frameName,'/Date');

threshAC = 0.5;
load stokeData_cdts2

for ii = 1:size(cd_1D,1)
    if rem(ii,10)==0
        ii
    end
    this_cd = cd_1D(ii,:);
	this_cdTSmooth = cdTSmooth_1D(ii,:);
	this_cdAPS = cdAPS_1D(ii,:);
    this_cdFilt = cdFilt_1D(ii,:);
    
    thisCD = this_cdAPS;
    ac = autocorr(thisCD,lagAC);
    arrayAC(ii) =  abs(ac(lagAC));
    %> threshAC;
end

arrayACInd = arrayAC>threshAC;

this_cdFiltInd = this_cdFilt(arrayACInd,:);
save arrayAC arrayAC this_cdFiltInd 

