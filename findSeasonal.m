load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';

dateAll = h5read(frameName,'/Date');

theseDates = datenum(num2str(dateAll),'yyyymmdd');
days = theseDates(end)-theseDates(end-1);
interpDates = theseDates(1):days:theseDates(end);


interpOut = interp1(theseDates,thesePoints, interpDates);


start = [41809 4012 1];
count = [44381-41809 4537-4012 170];


cd1 = h5read(frameName,'/Cumulative_Displacement',start,count);
cdts = h5read(frameName,'/Cumulative_Displacement_TSmooth',start,count);
lon1 = h5read(frameName,'/Longitude', start(1:2), count(1:2));
lat1 = h5read(frameName,'/Latitude', start(1:2), count(1:2));
cd1_100 = cd1(:,:,100);
thisInd2 = isnan(cd1_100);
lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);
for ii = 2:170
	this_cdts = cdts(:,:,ii);
	this_cd1 = cd1(:,:,ii);
	thisInd0 = isnan(this_cdts);
	thisInd1 = isnan(this_cd1);
    
    for iii = 1:length(thisInd0)
        thesePoints = thisInd0(:);
    end
	this_cdts_1D(:,ii-1)= this_cdts(~thisInd0);
	this_cd1_1D(:,ii-1)= this_cd1(~thisInd1);
end	

