load  UKEQs2016-2018
frameName = '030A_03647_101313-vel.h5';




dateAll = h5read(frameName,'/Date');

start = [41809 4012 1];
count = [44381-41809 4537-4012 length(dateAll)];
theseDates = datenum(num2str(dateAll),'yyyymmdd');
days = theseDates(end)-theseDates(end-1);
interpDates = theseDates(1):days:theseDates(end); interpDates  =interpDates';

cd = h5read(frameName,'/Cumulative_Displacement',start,count);
cdTSmooth = h5read(frameName,'/Cumulative_Displacement_TSmooth',start,count);
cdAPS = h5read(frameName,'/Cumulative_Displacement_APS',start,count);
cdFilt = h5read(frameName,'/Cumulative_Displacement_Filt',start,count);
lon1 = h5read(frameName,'/Longitude', start(1:2), count(1:2));
lat1 = h5read(frameName,'/Latitude', start(1:2), count(1:2));
cd1_100 = cd(:,:,100);
thisInd2 = isnan(cd1_100);
lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);


for ii = 1:size(cdTSmooth,1)
    ii
    for jj = 1:size(cdTSmooth,2)
        thisTScd = cd(ii,jj,:); thisTScd = thisTScd(:);
        thisTScdTSmooth = cdTSmooth(ii,jj,:); thisTScdTSmooth = thisTScdTSmooth(:);
        thisTScdAPS = cdAPS(ii,jj,:); thisTScdAPS = thisTScdAPS(:);
        thisTScdFilt = cdFilt(ii,jj,:); thisTScdFilt = thisTScdFilt(:);
        outcd(ii,jj,:) = interp1(theseDates,thisTScd, interpDates);
        outcdTSmooth(ii,jj,:) = interp1(theseDates,thisTScdTSmooth, interpDates);
        outcdAPS(ii,jj,:) = interp1(theseDates,thisTScdAPS, interpDates);
        outcdFilt(ii,jj,:) = interp1(theseDates,thisTScdFilt, interpDates);
    end
end

for ii = 1:size(outcd,3)
    this_cd = outcd(:,:,ii);
	this_cdTSmooth = outcdTSmooth(:,:,ii);
	this_cdAPS = outcdAPS(:,:,ii);
    this_cdFilt = outcdFilt(:,:,ii);
    
	thisInd = isnan(this_cdTSmooth);
    
    this_cd  = this_cd(~thisInd);
    this_cdTSmooth = this_cdTSmooth(~thisInd);
    this_cdAPS = this_cdAPS(~thisInd);
    this_cdFilt = this_cdFilt(~thisInd);
    
    cd_1D(:,ii)= this_cd(:);
	cdTSmooth_1D(:,ii)= this_cdTSmooth(:);
	cdAPS_1D(:,ii) = this_cdAPS(:);
    cdFilt_1D(:,ii) = this_cdFilt(:);
end	

save stokeData_cdts2 lon2 lat2 cd_1D cdTSmooth_1D cdAPS_1D cdFilt_1D
