%[41809:44381,4012:4537];
start = [41809 4012 1];
count = [44381-41809 4537-4012 170]


cd1 = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement',start,count);
cdts = h5read('030A_03647_101313-vel.h5','/Cumulative_Displacement_TSmooth',start,count);
lon1 = h5read('030A_03647_101313-vel.h5','/Longitude', start(1:2), count(1:2));
lat1 = h5read('030A_03647_101313-vel.h5','/Latitude', start(1:2), count(1:2));
cd1_100 = cd1(:,:,100);
thisInd2 = isnan(cd1_100);
lon2 = lon1(~thisInd2);
lat2 = lat1(~thisInd2);
for ii = 2:170
	this_cdts = cdts(:,:,ii);
	this_cd1 = cd1(:,:,ii);
	thisInd0 = isnan(this_cdts);
	thisInd1 = isnan(this_cd1);
	this_cdts_1D(:,ii-1)= this_cdts(~thisInd0);
	this_cd1_1D(:,ii-1)= this_cd1(~thisInd1);
end	
save stokeData_cdts lon2 lat2 this_cdts_1D this_cd1_1D
%plot(lt2,ln2,p4,'+');
%plot([lt2 ln2],p4,'+');
%plot(lt2,ln2,'+');
%scatter(lt2,ln2,p4);
%scatter3(lt2,ln2,p4);
%-- 12/04/19  4:07:26 pm BST --%
%edit
